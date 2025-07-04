#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import torch
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from tqdm import tqdm
import glob
import warnings
from scipy.spatial.transform import Rotation
import evo
from evo.core import trajectory
from evo.tools import file_interface
from evo.core import metrics
from evo.core import sync
from evo.tools import plot
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Use local imports instead of exps
try:
    from dataset import SCAREDRAWDataset
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.append('../')
    from dataset import SCAREDRAWDataset

from torch.utils.data import DataLoader
try:
    from .networks.resnet_encoder import AttentionalResnetEncoder, MultiHeadAttentionalResnetEncoder, ResnetEncoder
    from .networks.pose_decoder import PoseDecoder_with_intrinsics_old as PoseDecoder_i
except ImportError:
    from networks.resnet_encoder import AttentionalResnetEncoder, MultiHeadAttentionalResnetEncoder, ResnetEncoder
    from networks.pose_decoder import PoseDecoder_with_intrinsics_old as PoseDecoder_i

try:
    from layers import transformation_from_parameters
except ImportError:
    from .layers import transformation_from_parameters
try:
    from utils import readlines
except ImportError:
    from .utils import readlines
try:
    from options import MonodepthOptions
except ImportError:
    try:
        from ..options import MonodepthOptions
    except ImportError:
        # Create a placeholder class if options is not available
        class MonodepthOptions:
            pass

# Constants that were imported from exp_setup_local
platform = 'local' if os.getcwd().startswith('/mnt') else 'cluster'
ds_base = '/mnt/c/Users/14152/ZCH/Dev/datasets' if platform == 'local' else '/cluster/project7/Llava_2024/changhao/datasets'

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- POSE EVALUATION FUNCTIONS -----

def transformation_matrix_to_kitti_format(transformations):
    """Convert transformation matrices to KITTI format (r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3)"""
    kitti_data = []
    
    for transformation in transformations:
        # Extract rotation matrix and translation vector
        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        # Format: r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
        line = " ".join([str(rotation_matrix[0, 0]), str(rotation_matrix[0, 1]), str(rotation_matrix[0, 2]), str(translation[0]),
                         str(rotation_matrix[1, 0]), str(rotation_matrix[1, 1]), str(rotation_matrix[1, 2]), str(translation[1]),
                         str(rotation_matrix[2, 0]), str(rotation_matrix[2, 1]), str(rotation_matrix[2, 2]), str(translation[2])])
        kitti_data.append(line)
        
    return kitti_data

def transformation_matrix_to_tum_format(transformations):
    """Convert transformation matrices to TUM format (timestamp tx ty tz qx qy qz qw)"""
    tum_data = []
    timestamp = 0
    
    for transformation in transformations:
        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        # Convert rotation matrix to quaternion
        r = Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()  # x, y, z, w format
        
        # TUM format uses quaternion in w, x, y, z order
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        
        # Format: timestamp tx ty tz qx qy qz qw
        tum_data.append(f"{timestamp} {translation[0]} {translation[1]} {translation[2]} {qx} {qy} {qz} {qw}")
        timestamp += 0.1  # Increment timestamp
        
    return tum_data

def write_trajectory_to_file(transformation_matrices, output_path, format='kitti'):
    """Write transformation matrices to a file in specified format"""
    if format == 'tum':
        data = transformation_matrix_to_tum_format(transformation_matrices)
    else:  # kitti format
        data = transformation_matrix_to_kitti_format(transformation_matrices)
    
    with open(output_path, 'w') as f:
        for line in data:
            f.write(line + '\n')
    
    return output_path

def load_kitti_poses(pose_file_path):
    """Load KITTI pose file and convert to evo trajectory"""
    poses = []
    with open(pose_file_path, 'r') as f:
        for line in f.readlines():
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            poses.append(pose)
    
    if len(poses) == 0:
        raise ValueError(f"No poses found in file: {pose_file_path}")
    
    # Pre-allocate arrays for trajectory
    positions = np.zeros((len(poses), 3))
    orientations = np.zeros((len(poses), 4))
    timestamps = np.zeros(len(poses))
    
    # Fill arrays with pose data
    for i, pose in enumerate(poses):
        # Extract translation
        positions[i] = pose[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        r = Rotation.from_matrix(pose[:3, :3])
        q = r.as_quat()  # x, y, z, w
        orientations[i] = np.array([q[3], q[0], q[1], q[2]])  # convert to w, x, y, z
        timestamps[i] = float(i)
    
    # Create trajectory with all data at once
    traj = trajectory.PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations, 
        timestamps=timestamps
    )
    
    return traj

# ----- INTRINSICS EVALUATION FUNCTIONS -----

def compute_intrinsic_errors(pred_intrinsics, gt_intrinsics):
    """Compute scale-invariant errors between predicted and ground truth intrinsics matrices.
    
    Args:
        pred_intrinsics: Predicted intrinsics matrix or batch of matrices (B, 4, 4)
        gt_intrinsics: Ground truth intrinsics matrix (4, 4)
        
    Returns:
        Dictionary containing scale-invariant error metrics
    """
    # If pred_intrinsics is a batch, compute mean errors
    if len(pred_intrinsics.shape) > 2:
        errors = {
            'scale_factor': [],
            'scale_adjusted_fx_abs_error': [],
            'scale_adjusted_fy_abs_error': [],
            'scale_adjusted_cx_abs_error': [],
            'scale_adjusted_cy_abs_error': [],
            'scale_adjusted_fx_rel_error': [],
            'scale_adjusted_fy_rel_error': [],
            'scale_adjusted_cx_rel_error': [],
            'scale_adjusted_cy_rel_error': [],
            'scale_adjusted_matrix_frobenius_norm': []
        }
        
        for pred_mat in pred_intrinsics:
            # Extract intrinsic parameters
            pred_fx = pred_mat[0, 0].item()
            pred_fy = pred_mat[1, 1].item()
            pred_cx = pred_mat[0, 2].item()
            pred_cy = pred_mat[1, 2].item()
            
            gt_fx = gt_intrinsics[0, 0]
            gt_fy = gt_intrinsics[1, 1]
            gt_cx = gt_intrinsics[0, 2]
            gt_cy = gt_intrinsics[1, 2]
            
            # Calculate optimal scale factor based on focal length ratio
            scale_fx = gt_fx / pred_fx if pred_fx != 0 else 1.0
            scale_fy = gt_fy / pred_fy if pred_fy != 0 else 1.0
            
            # Use the average scale
            scale = (scale_fx + scale_fy) / 2.0
            errors['scale_factor'].append(scale)
            
            # Apply the scale to the predicted intrinsics
            scaled_pred_fx = pred_fx * scale
            scaled_pred_fy = pred_fy * scale
            # Principal points also scale the same way as focal lengths in monocular depth estimation
            scaled_pred_cx = pred_cx * scale
            scaled_pred_cy = pred_cy * scale
            
            # Compute scale-adjusted absolute errors
            errors['scale_adjusted_fx_abs_error'].append(abs(scaled_pred_fx - gt_fx))
            errors['scale_adjusted_fy_abs_error'].append(abs(scaled_pred_fy - gt_fy))
            errors['scale_adjusted_cx_abs_error'].append(abs(scaled_pred_cx - gt_cx))
            errors['scale_adjusted_cy_abs_error'].append(abs(scaled_pred_cy - gt_cy))
            
            # Compute scale-adjusted relative errors (%)
            errors['scale_adjusted_fx_rel_error'].append(abs(scaled_pred_fx - gt_fx) / gt_fx * 100)
            errors['scale_adjusted_fy_rel_error'].append(abs(scaled_pred_fy - gt_fy) / gt_fy * 100)
            errors['scale_adjusted_cx_rel_error'].append(abs(scaled_pred_cx - gt_cx) / gt_cx * 100)
            errors['scale_adjusted_cy_rel_error'].append(abs(scaled_pred_cy - gt_cy) / gt_cy * 100)
            
            # Scale-adjusted Frobenius norm
            scaled_pred_mat = pred_mat.clone()
            scaled_pred_mat[0, 0] *= scale  # fx
            scaled_pred_mat[1, 1] *= scale  # fy
            scaled_pred_mat[0, 2] *= scale  # cx
            scaled_pred_mat[1, 2] *= scale  # cy
            diff_mat_scaled = scaled_pred_mat[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
            errors['scale_adjusted_matrix_frobenius_norm'].append(np.linalg.norm(diff_mat_scaled, 'fro'))
        
        # Calculate statistics
        result = {}
        for key, values in errors.items():
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
            result[f'{key}_median'] = np.median(values)
            result[f'{key}_min'] = np.min(values)
            result[f'{key}_max'] = np.max(values)
        
        # Add raw errors for plotting
        result['errors'] = errors
        
        return result
    else:
        # Single prediction case
        pred_fx = pred_intrinsics[0, 0].item()
        pred_fy = pred_intrinsics[1, 1].item()
        pred_cx = pred_intrinsics[0, 2].item()
        pred_cy = pred_intrinsics[1, 2].item()
        
        gt_fx = gt_intrinsics[0, 0]
        gt_fy = gt_intrinsics[1, 1]
        gt_cx = gt_intrinsics[0, 2]
        gt_cy = gt_intrinsics[1, 2]
        
        # Calculate optimal scale factor based on focal length ratio
        scale_fx = gt_fx / pred_fx if pred_fx != 0 else 1.0
        scale_fy = gt_fy / pred_fy if pred_fy != 0 else 1.0
        
        # Use the average scale
        scale = (scale_fx + scale_fy) / 2.0
        
        # Apply the scale to the predicted intrinsics
        scaled_pred_fx = pred_fx * scale
        scaled_pred_fy = pred_fy * scale
        # Principal points scale the same way as focal lengths
        scaled_pred_cx = pred_cx * scale
        scaled_pred_cy = pred_cy * scale
        
        # Compute scale-adjusted absolute errors
        scale_adjusted_fx_abs_error = abs(scaled_pred_fx - gt_fx)
        scale_adjusted_fy_abs_error = abs(scaled_pred_fy - gt_fy)
        scale_adjusted_cx_abs_error = abs(scaled_pred_cx - gt_cx)
        scale_adjusted_cy_abs_error = abs(scaled_pred_cy - gt_cy)
        
        # Compute scale-adjusted relative errors (%)
        scale_adjusted_fx_rel_error = abs(scaled_pred_fx - gt_fx) / gt_fx * 100
        scale_adjusted_fy_rel_error = abs(scaled_pred_fy - gt_fy) / gt_fy * 100
        scale_adjusted_cx_rel_error = abs(scaled_pred_cx - gt_cx) / gt_cx * 100
        scale_adjusted_cy_rel_error = abs(scaled_pred_cy - gt_cy) / gt_cy * 100
        
        # Scale-adjusted Frobenius norm
        scaled_pred_mat = pred_intrinsics.clone()
        scaled_pred_mat[0, 0] *= scale  # fx
        scaled_pred_mat[1, 1] *= scale  # fy
        scaled_pred_mat[0, 2] *= scale  # cx
        scaled_pred_mat[1, 2] *= scale  # cy
        diff_mat_scaled = scaled_pred_mat[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
        scale_adjusted_matrix_frobenius_norm = np.linalg.norm(diff_mat_scaled, 'fro')
        
        return {
            'scale_factor': scale,
            'scale_adjusted_fx_abs_error': scale_adjusted_fx_abs_error,
            'scale_adjusted_fy_abs_error': scale_adjusted_fy_abs_error,
            'scale_adjusted_cx_abs_error': scale_adjusted_cx_abs_error,
            'scale_adjusted_cy_abs_error': scale_adjusted_cy_abs_error,
            'scale_adjusted_fx_rel_error': scale_adjusted_fx_rel_error,
            'scale_adjusted_fy_rel_error': scale_adjusted_fy_rel_error,
            'scale_adjusted_cx_rel_error': scale_adjusted_cx_rel_error,
            'scale_adjusted_cy_rel_error': scale_adjusted_cy_rel_error,
            'scale_adjusted_matrix_frobenius_norm': scale_adjusted_matrix_frobenius_norm
        }

def visualize_intrinsics_errors(errors, output_dir, name_prefix=""):
    """Visualize intrinsics errors as boxplots and histograms.
    
    Args:
        errors: Dictionary containing error values from compute_intrinsic_errors
        output_dir: Directory to save visualization plots
        name_prefix: Prefix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_errors = errors['errors']
    
    # Create parameter names and descriptions
    param_names = ['fx', 'fy', 'cx', 'cy']
    param_desc = ['Focal Length X', 'Focal Length Y', 'Principal Point X', 'Principal Point Y']
    
    # Plot scale-adjusted absolute errors
    plt.figure(figsize=(12, 8))
    scale_abs_data = [raw_errors['scale_adjusted_fx_abs_error'], raw_errors['scale_adjusted_fy_abs_error'], 
                      raw_errors['scale_adjusted_cx_abs_error'], raw_errors['scale_adjusted_cy_abs_error']]
    plt.boxplot(scale_abs_data, labels=param_names)
    plt.title('Scale-Adjusted Absolute Errors in Camera Intrinsic Parameters')
    plt.ylabel('Scale-Adjusted Absolute Error (pixels)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_adjusted_absolute_errors_boxplot.png"))
    plt.close()
    
    # Plot scale-adjusted relative errors
    plt.figure(figsize=(12, 8))
    scale_rel_data = [raw_errors['scale_adjusted_fx_rel_error'], raw_errors['scale_adjusted_fy_rel_error'], 
                      raw_errors['scale_adjusted_cx_rel_error'], raw_errors['scale_adjusted_cy_rel_error']]
    plt.boxplot(scale_rel_data, labels=param_names)
    plt.title('Scale-Adjusted Relative Errors in Camera Intrinsic Parameters')
    plt.ylabel('Scale-Adjusted Relative Error (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_adjusted_relative_errors_boxplot.png"))
    plt.close()
    
    # Plot scale factors
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_errors['scale_factor'], kde=True)
    plt.title('Distribution of Scale Factors')
    plt.xlabel('Scale Factor')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_factors.png"))
    plt.close()
    
    # Create histograms for each parameter
    for i, param in enumerate(param_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scale-adjusted absolute error histogram
        sns.histplot(scale_abs_data[i], kde=True, ax=ax1)
        ax1.set_title(f'Scale-Adjusted Absolute Error - {param_desc[i]}')
        ax1.set_xlabel('Scale-Adjusted Absolute Error (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Scale-adjusted relative error histogram
        sns.histplot(scale_rel_data[i], kde=True, ax=ax2)
        ax2.set_title(f'Scale-Adjusted Relative Error - {param_desc[i]}')
        ax2.set_xlabel('Scale-Adjusted Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name_prefix}{param}_error_distribution.png"))
        plt.close()
    
    # Matrix norm error histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_errors['scale_adjusted_matrix_frobenius_norm'], kde=True)
    plt.title('Scale-Adjusted Frobenius Norm of Intrinsics Error Matrix')
    plt.xlabel('Scale-Adjusted Frobenius Norm')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}matrix_norm_error.png"))
    plt.close()

# ----- POSE EVALUATION FUNCTIONS -----

def dump_trajectory_poses(source_to_target_transformations):
    """Convert relative transformations to absolute camera poses
    
    Args:
        source_to_target_transformations: Array of relative transformation matrices
        
    Returns:
        List of absolute transformation matrices
    """
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms

def compute_trajectory_scale(gtruth, pred):
    """Compute optimal scale factor between ground truth and predicted trajectories
    
    Args:
        gtruth: Ground truth absolute poses
        pred: Predicted absolute poses
        
    Returns:
        Optimal scale factor
    """
    # Ensure both arrays have the same length
    min_len = min(len(gtruth), len(pred))
    gtruth = gtruth[:min_len]
    pred = pred[:min_len]
    
    # Extract translation components
    gt_translations = gtruth[:, :3, 3]
    pred_translations = pred[:, :3, 3]
    
    # Optimize the scaling factor
    numerator = np.sum(gt_translations * pred_translations)
    denominator = np.sum(pred_translations ** 2)
    
    if denominator == 0:
        return 1.0  # Avoid division by zero
    
    scale = numerator / denominator
    return scale

def plot_3d_trajectory_comparison(pred_poses, gt_poses, output_path, dataset_name, pose_seq):
    """Generate 3D trajectory comparison plot matching the user's code format
    
    Args:
        pred_poses: Array of predicted relative transformation matrices
        gt_poses: Array of ground truth relative transformation matrices  
        output_path: Path to save the plot
        dataset_name: Name of the dataset
        pose_seq: Sequence number
    """
    print(f"Generating 3D trajectory comparison for {dataset_name} seq {pose_seq}")
    print(f"Predicted poses shape: {pred_poses.shape}")
    print(f"Ground truth poses shape: {gt_poses.shape}")
    
    # Convert relative poses to absolute poses
    dump_gt = np.array(dump_trajectory_poses(gt_poses))
    dump_pred = np.array(dump_trajectory_poses(pred_poses))
    
    print(f"Absolute GT poses shape: {dump_gt.shape}")
    print(f"Absolute pred poses shape: {dump_pred.shape}")
    
    # Ensure both trajectories have the same length by taking the minimum
    min_len = min(len(dump_gt), len(dump_pred))
    dump_gt = dump_gt[:min_len]
    dump_pred = dump_pred[:min_len]
    
    print(f"Using trajectory length: {min_len}")
    
    # Apply scale correction using the same method as reference code
    try:
        scale = compute_trajectory_scale(dump_gt, dump_pred)
        print(f"Computed scale factor: {scale}")
    except Exception as e:
        print(f"Error computing scale: {e}")
        scale = 1.0  # Fallback to no scaling
    
    scale_pred = dump_pred * scale
    
    # Extract trajectory points (following reference code style)
    num = min_len
    points_pred = []
    points_gt = []
    origin = np.array([[0], [0], [0], [1]])
    
    for i in range(0, num):
        point_pred = np.dot(scale_pred[i], origin)
        point_gt = np.dot(dump_gt[i], origin)
        
        points_pred.append(point_pred)
        points_gt.append(point_gt)
    
    points_pred = np.array(points_pred)
    points_gt = np.array(points_gt)
    
    # Create 3D figure (following reference code style)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # # Set 3D plot background to white
    # ax.xaxis.pane.fill = True
    # ax.yaxis.pane.fill = True
    # ax.zaxis.pane.fill = True
    # ax.xaxis.pane.set_facecolor('white')
    # ax.yaxis.pane.set_facecolor('white')
    # ax.zaxis.pane.set_facecolor('white')
    # ax.xaxis.pane.set_alpha(1.0)
    # ax.yaxis.pane.set_alpha(1.0)
    # ax.zaxis.pane.set_alpha(1.0)
    
    # Set black grid
    ax.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.8)
    
    # Set axis line colors to black
    ax.xaxis._axinfo['grid']['color'] = 'aeaeae'
    ax.yaxis._axinfo['grid']['color'] = 'aeaeae'  
    ax.zaxis._axinfo['grid']['color'] = 'aeaeae'

    
    # Set figure information (following reference code style)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    
    # Draw the trajectories (following reference code style)
    figure1, = ax.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], 
                      label='GT', linestyle='-', c='b', linewidth=1.6)
    figure2, = ax.plot(points_pred[:, 0, 0], points_pred[:, 1, 0], points_pred[:, 2, 0], 
                      label='Prediction', linestyle='-', c='g', linewidth=1.6)
    
    plt.legend()
    plt.plot()
    plt.savefig(output_path, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"3D trajectory comparison saved to: {output_path}")

# ----- UNIFIED EVALUATION FUNCTION -----

def evaluate(opt, load_weights_folder, dataset_name="SCARED", pose_seq=1, 
             evaluate_pose=True, evaluate_intrinsics=True, 
             gt_path=None, filename_list_path=None,
             pose_encoder_class=MultiHeadAttentionalResnetEncoder, pose_decoder_class=PoseDecoder_i):
    """Unified evaluation function for both pose and intrinsics
    
    Args:
        opt: Options object containing configuration
        load_weights_folder: Path to folder containing model weights
        dataset_name: Name of the dataset for result reporting
        pose_seq: Pose sequence number (if applicable)
        evaluate_pose: Whether to evaluate pose
        evaluate_intrinsics: Whether to evaluate intrinsics
        gt_path: Path to ground truth poses (for pose evaluation)
        filename_list_path: Path to file listing test image filenames
        
    Returns:
        Dictionary with evaluation results
    """
    # 获取数据集路径 / Get dataset path
    # Use the same ds_base as defined above
    if dataset_name == 'SCARED':
        ds_path = os.path.join(ds_base, 'SCARED_Images_Resized')
    elif dataset_name in ['SyntheticColon', 'C3VD']:
        ds_path = os.path.join(ds_base, f'{dataset_name}_as_SCARED')
    elif dataset_name in ['test_time_train_1', 'test_time_train_2']:
        ds_path = os.path.join(ds_base, f'C3VD_{dataset_name}')
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    assert os.path.isdir(load_weights_folder), \
        f"Cannot find a folder at {load_weights_folder}"
    
    # Setup dataset and dataloader
    if filename_list_path is None:
        if dataset_name == 'SCARED':
            filename_list_path = os.path.join(ds_path, "splits", f"test_files_sequence{pose_seq}.txt")
            filenames = readlines(filename_list_path)
        elif dataset_name in ['SyntheticColon', 'C3VD', 'test_time_train_1', 'test_time_train_2']:
            # 默认使用第一个test folder / Use the first test folder by default
            test_folders_file = os.path.join(ds_path, 'splits', 'test_folders.txt')
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            test_folder = test_folders[0].strip()
            # 读取这个文件夹下的所有图片，按编号顺序而非字典序 / Read all images in this folder, sorted by number order instead of alphabetical order
            test_folder_path = os.path.join(ds_path, test_folder, 'image_02', 'data')
            filenames = sorted(os.listdir(test_folder_path), key=lambda x: int(x.split('.')[0]))
            # Extract the last two parts of the test folder path
            folder_parts = test_folder.split('/')
            if len(folder_parts) >= 2:
                folder_prefix = '/'.join(folder_parts[-2:])
            else:
                folder_prefix = test_folder

            # Format filenames as "dataset2/keyframe1 369 l"
            filenames = [f"{folder_prefix} {f.split('.')[0]} l" for f in filenames]
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    print(f"Using test files list: {filename_list_path}")
    
    
    dataset = SCAREDRAWDataset(ds_path, filenames, opt.height, opt.width,
                               [0, 1], 1, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    # Load models
    pose_encoder_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(load_weights_folder, "pose.pth")
    
    # Load encoder
    pose_encoder = pose_encoder_class(opt.num_layers, False, num_input_images=2)
    static_dict = torch.load(pose_encoder_path, map_location=device.type)
    if 'module.' in list(static_dict.keys())[0]:
        static_dict = {k.replace('module.', ''): v for k, v in static_dict.items()}
    pose_encoder.load_state_dict(static_dict)
    
    # Load decoder with intrinsics prediction
    pose_decoder = pose_decoder_class(
        pose_encoder.num_ch_enc, 
        image_width=opt.width, 
        image_height=opt.height, 
        predict_intrinsics=opt.learn_intrinsics,
        simplified_intrinsic=opt.simplified_intrinsic,
        num_input_features=1, 
        num_frames_to_predict_for=2
    )
    pose_decoder_dict = torch.load(pose_decoder_path, map_location=device.type)
    if 'module.' in list(pose_decoder_dict.keys())[0]:
        pose_decoder_dict = {k.replace('module.', ''): v for k, v in pose_decoder_dict.items()}
    pose_decoder.load_state_dict(pose_decoder_dict)
    
    # Move models to device and set to eval mode
    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()
    
    # Collect predictions in a single pass
    pred_poses = []
    all_pred_intrinsics = []
    
    opt.frame_ids = [0, 1]  # pose network only takes two frames as input
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Running inference"):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
            
            # Prepare inputs and run the model
            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)
            features = [pose_encoder(all_color_aug)]
            
            # Get predictions (axisangle, translation, intrinsics)
            axisangle, translation, intrinsics = pose_decoder(features)
            
            # Save predicted poses
            if evaluate_pose:
                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            
            # Save predicted intrinsics
            if evaluate_intrinsics:
                all_pred_intrinsics.append(intrinsics)
    
    results = {}
    
    # ----- POSE EVALUATION -----
    if evaluate_pose:
        pred_poses = np.concatenate(pred_poses)
        
        # Create temporary files for evaluation
        temp_dir = os.path.join(os.path.dirname(load_weights_folder), "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)
        pred_traj_file = os.path.join(temp_dir, f"{dataset_name}_{pose_seq}_pred_trajectory.txt")
        
        # Convert predicted poses to trajectory file
        accumulated_poses = [np.eye(4)]
        for i in range(len(pred_poses)):
            accumulated_poses.append(np.dot(accumulated_poses[-1], pred_poses[i]))
        
        # Write trajectory in KITTI format
        write_trajectory_to_file(accumulated_poses[1:], pred_traj_file, format='kitti')
        
        # Define ground truth path if not provided
        if gt_path is None:
            if dataset_name == 'SCARED':
                gt_path = os.path.join(ds_path, "splits", f"traj_kitti_{pose_seq}.txt")
            else:
                # 从 splits/test_folders.txt 中读取 ground truth 路径 / Read ground truth path from splits/test_folders.txt
                gt_folders_file = os.path.join(ds_path, 'splits', 'test_folders.txt')
                with open(gt_folders_file, 'r') as f:
                    gt_folders = f.readlines()
                gt_folder = gt_folders[pose_seq - 1].strip()
                gt_path = os.path.join(ds_path, gt_folder, 'traj_kitti.txt')
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth pose file not found at {gt_path}")
            print("Skipping pose evaluation")
        else:
            # Load trajectories with evo
            print("Loading and processing trajectories")
            traj_ref = load_kitti_poses(gt_path)
            traj_est = load_kitti_poses(pred_traj_file)
            
            # Associate, register and align trajectories
            print("Registering and aligning trajectories")
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
            traj_est.align(traj_ref, correct_scale=True)
            
            # Calculate ATE
            ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
            ate_metric.process_data((traj_ref, traj_est))
            ate_statistics = ate_metric.get_all_statistics()
            
            # Calculate RPE
            rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)
            rpe_metric.process_data((traj_ref, traj_est))
            rpe_statistics = rpe_metric.get_all_statistics()
            
            # Save trajectory visualization
            # Create output folder for visualizations
            vis_output_dir = os.path.join(os.path.dirname(load_weights_folder), "trajectory_vis")
            os.makedirs(vis_output_dir, exist_ok=True)
            
            # Create a plot collection for visualization
            plot_collection = plot.PlotCollection(f"Trajectory Evaluation - {dataset_name} {pose_seq}")
            
            # Plot ATE error
            fig_ate = plt.figure(figsize=(10, 8))
            plot.error_array(fig_ate.gca(), ate_metric.error, statistics=ate_statistics,
                           name="APE", title=str(ate_metric))
            plot_collection.add_figure("ate_error", fig_ate)
            
            # Plot trajectory with ATE error colormap
            fig_traj_ate = plt.figure(figsize=(10, 8))
            plot_mode = plot.PlotMode.xy
            ax = plot.prepare_axis(fig_traj_ate, plot_mode)
            plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
            plot.traj_colormap(ax, traj_est, ate_metric.error, plot_mode,
                             min_map=ate_statistics["min"],
                             max_map=ate_statistics["max"],
                             title="ATE mapped onto trajectory")
            plot_collection.add_figure("traj_ate", fig_traj_ate)
            
            # Plot RPE error
            fig_rpe = plt.figure(figsize=(10, 8))
            plot.error_array(fig_rpe.gca(), rpe_metric.error, statistics=rpe_statistics,
                           name="RPE", title=str(rpe_metric))
            plot_collection.add_figure("rpe_error", fig_rpe)
            
            # Plot 3D view of trajectories
            fig_3d = plt.figure(figsize=(10, 8))
            ax = plot.prepare_axis(fig_3d, plot.PlotMode.xyz)
            plot.traj(ax, plot.PlotMode.xyz, traj_ref, '--', 'gray', 'reference')
            plot.traj(ax, plot.PlotMode.xyz, traj_est, '-', 'blue', 'estimated')
            ax.legend()
            plot_collection.add_figure("traj_3d", fig_3d)
            
            # Save all plots to the output directory
            output_filename = os.path.join(vis_output_dir, f"{dataset_name}_seq{pose_seq}")
            plot_collection.export(output_filename, confirm_overwrite=False)
            plt.close('all')
            
            # Generate additional 3D trajectory comparison plot (user's format)
            # Load ground truth absolute poses from KITTI file
            gt_absolute_poses_list = []
            with open(gt_path, 'r') as f:
                for line in f.readlines():
                    values = [float(v) for v in line.strip().split()]
                    pose = np.eye(4)
                    pose[0, 0:4] = values[0:4]
                    pose[1, 0:4] = values[4:8]
                    pose[2, 0:4] = values[8:12]
                    gt_absolute_poses_list.append(pose)
            
            # Convert absolute poses back to relative poses for gt
            gt_relative_poses = []
            for i in range(1, len(gt_absolute_poses_list)):
                relative_pose = np.linalg.inv(gt_absolute_poses_list[i-1]) @ gt_absolute_poses_list[i]
                gt_relative_poses.append(relative_pose)
            
            gt_relative_poses = np.array(gt_relative_poses)
            
            # Generate 3D trajectory comparison plot
            trajectory_3d_path = os.path.join(vis_output_dir, f'trajectory_pose_seq{pose_seq}.png')
            plot_3d_trajectory_comparison(pred_poses, gt_relative_poses, 
                                        trajectory_3d_path, dataset_name, pose_seq)
            
            # Print results
            print(f"ATE RMSE: {ate_statistics['rmse']}, std: {ate_statistics['std']}")
            print(f"RPE mean: {rpe_statistics['mean']}, std: {rpe_statistics['std']}")
            
            # Save results to return dictionary
            results['ate_rmse'] = ate_statistics['rmse']
            results['ate_std'] = ate_statistics['std']
            results['rpe_mean'] = rpe_statistics['mean']
            results['rpe_std'] = rpe_statistics['std']
    
    # ----- INTRINSICS EVALUATION -----
    if evaluate_intrinsics:
        # Load ground truth intrinsics
        gt_intrinsics_path = os.path.join(ds_path, "intrinsics.txt")
        if not os.path.exists(gt_intrinsics_path):
            print(f"Warning: Ground truth intrinsics file not found at {gt_intrinsics_path}")
            print("Looking for K.txt instead...")
            gt_intrinsics_path = os.path.join(ds_path, "K.txt")
        
        if not os.path.exists(gt_intrinsics_path):
            print(f"Warning: Cannot find ground truth intrinsics at {gt_intrinsics_path} or K.txt")
            print("Skipping intrinsics evaluation")
        else:
            # Load ground truth intrinsics
            gt_intrinsics = np.loadtxt(gt_intrinsics_path)
            
            # Check if intrinsics is just 3x3 and convert to 4x4 if needed
            if gt_intrinsics.shape == (3, 3):
                temp = np.eye(4)
                temp[:3, :3] = gt_intrinsics
                gt_intrinsics = temp
            
            print(f"Loaded ground truth intrinsics:\n{gt_intrinsics}")
            
            # Concatenate all predictions
            all_pred_intrinsics = torch.cat(all_pred_intrinsics, dim=0)
            
            # Compute error metrics
            print(f"\nComputing error metrics for {all_pred_intrinsics.shape[0]} predictions...")
            error_metrics = compute_intrinsic_errors(all_pred_intrinsics, gt_intrinsics)
            
            # Create visualization directory
            vis_dir = os.path.join(os.path.dirname(load_weights_folder), "intrinsics_vis")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Visualize errors
            prefix = f"{dataset_name}_{pose_seq}_"
            visualize_intrinsics_errors(error_metrics, vis_dir, prefix)
            
            # Calculate mean predicted intrinsics for better comparison
            mean_pred_intrinsics = torch.mean(all_pred_intrinsics, dim=0).cpu().numpy()
            
            # Print the mean predicted intrinsics and ground truth for comparison
            print("\nMean Predicted Intrinsics:")
            print(mean_pred_intrinsics)
            
            # Calculate and print the scale-adjusted mean predicted intrinsics
            scale = error_metrics['scale_factor_mean']
            scaled_pred_intrinsics = mean_pred_intrinsics.copy()
            scaled_pred_intrinsics[0, 0] *= scale  # fx
            scaled_pred_intrinsics[1, 1] *= scale  # fy
            scaled_pred_intrinsics[0, 2] *= scale  # cx
            scaled_pred_intrinsics[1, 2] *= scale  # cy
            
            print("\nScale-Adjusted Mean Predicted Intrinsics (scale factor: {:.4f}):".format(scale))
            print(np.array2string(scaled_pred_intrinsics, precision=2, suppress_small=True, separator=', '))
            
            # Print scale-invariant metrics summary
            print("\nScale-Invariant Intrinsics Error Summary:")
            print(f"Mean Scale Factor: {error_metrics['scale_factor_mean']:.4f} ± {error_metrics['scale_factor_std']:.4f}")
            print(f"Scale-Adjusted Focal Length X: {error_metrics['scale_adjusted_fx_abs_error_mean']:.2f} ± "
                f"{error_metrics['scale_adjusted_fx_abs_error_std']:.2f} pixels "
                f"({error_metrics['scale_adjusted_fx_rel_error_mean']:.2f}% relative)")
            print(f"Scale-Adjusted Focal Length Y: {error_metrics['scale_adjusted_fy_abs_error_mean']:.2f} ± "
                f"{error_metrics['scale_adjusted_fy_abs_error_std']:.2f} pixels "
                f"({error_metrics['scale_adjusted_fy_rel_error_mean']:.2f}% relative)")
            print(f"Scale-Adjusted Principal Point X: {error_metrics['scale_adjusted_cx_abs_error_mean']:.2f} ± "
                f"{error_metrics['scale_adjusted_cx_abs_error_std']:.2f} pixels "
                f"({error_metrics['scale_adjusted_cx_rel_error_mean']:.2f}% relative)")
            print(f"Scale-Adjusted Principal Point Y: {error_metrics['scale_adjusted_cy_abs_error_mean']:.2f} ± "
                f"{error_metrics['scale_adjusted_cy_abs_error_std']:.2f} pixels "
                f"({error_metrics['scale_adjusted_cy_rel_error_mean']:.2f}% relative)")
            print(f"Scale-Adjusted Matrix Frobenius Norm: {error_metrics['scale_adjusted_matrix_frobenius_norm_mean']:.2f} ± "
                f"{error_metrics['scale_adjusted_matrix_frobenius_norm_std']:.2f}")
            
            # Save summary to file
            summary_file = os.path.join(vis_dir, f"{prefix}summary.txt")
            with open(summary_file, "w") as f:
                f.write("Scale-Invariant Intrinsics Error Summary\n")
                f.write("====================================\n\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Pose Sequence: {pose_seq}\n")
                f.write(f"Model: {os.path.basename(load_weights_folder)}\n\n")
                
                f.write("Mean Predicted Intrinsics:\n")
                f.write(f"{mean_pred_intrinsics}\n\n")
                
                f.write("Ground Truth Intrinsics:\n")
                f.write(f"{gt_intrinsics}\n\n")
                
                f.write("Scale-Invariant Metrics:\n")
                f.write(f"Mean Scale Factor: {error_metrics['scale_factor_mean']:.4f} ± {error_metrics['scale_factor_std']:.4f}\n")
                f.write(f"Scale-Adjusted Focal Length X: {error_metrics['scale_adjusted_fx_abs_error_mean']:.2f} ± "
                      f"{error_metrics['scale_adjusted_fx_abs_error_std']:.2f} pixels "
                      f"({error_metrics['scale_adjusted_fx_rel_error_mean']:.2f}% relative)\n")
                f.write(f"Scale-Adjusted Focal Length Y: {error_metrics['scale_adjusted_fy_abs_error_mean']:.2f} ± "
                      f"{error_metrics['scale_adjusted_fy_abs_error_std']:.2f} pixels "
                      f"({error_metrics['scale_adjusted_fy_rel_error_mean']:.2f}% relative)\n")
                f.write(f"Scale-Adjusted Principal Point X: {error_metrics['scale_adjusted_cx_abs_error_mean']:.2f} ± "
                      f"{error_metrics['scale_adjusted_cx_abs_error_std']:.2f} pixels "
                      f"({error_metrics['scale_adjusted_cx_rel_error_mean']:.2f}% relative)\n")
                f.write(f"Scale-Adjusted Principal Point Y: {error_metrics['scale_adjusted_cy_abs_error_mean']:.2f} ± "
                      f"{error_metrics['scale_adjusted_cy_abs_error_std']:.2f} pixels "
                      f"({error_metrics['scale_adjusted_cy_rel_error_mean']:.2f}% relative)\n")
                f.write(f"Scale-Adjusted Matrix Frobenius Norm: {error_metrics['scale_adjusted_matrix_frobenius_norm_mean']:.2f} ± "
                      f"{error_metrics['scale_adjusted_matrix_frobenius_norm_std']:.2f}\n")
            
            # Add results to return dictionary
            results['intrinsics_error'] = error_metrics
            results['mean_pred_intrinsics'] = mean_pred_intrinsics
    
    return results

def evaluate_batch(opt, dataset_name, model_dir, debug=False):
    """Run batch evaluation on specified dataset and model
    
    Args:
        opt: Options object containing configuration
        dataset_name: Name of the dataset for evaluation
        model_dir: Directory containing model weights
        debug: Run in debug mode (limited evaluation)
    """
    # Create results directory
    results_dir = os.path.join(model_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # CSV file paths
    csv_file = os.path.join(results_dir, f'{dataset_name}_all_results.csv')
    best_csv_file = os.path.join(results_dir, f'{dataset_name}_best_results.csv')
    
    weight_folders = glob.glob(os.path.join(model_dir, 'weights_*'))
    weight_indices = [int(os.path.basename(folder).split('_')[1]) for folder in weight_folders]
    weight_indices.sort()
    
    if debug:
        weight_indices = weight_indices[:2]  # Only evaluate first two weight folders in debug mode
    
    # For tracking all results
    all_results = []
    best_results = []
    
    # Text summary file (keeping for backwards compatibility)
    result_file = os.path.join(results_dir, f'{dataset_name}_results.txt')
    with open(result_file, 'w') as f_result:
        f_result.write(f'Evaluation results for {dataset_name}:\n')
        
        if dataset_name == 'SCARED':
            # -- SCARED Traj 1 and 2 --
            for seq in [1, 2]:
                min_ate = np.inf
                min_re = np.inf
                ate_std = np.inf
                re_std = np.inf
                best_i = 0
                
                best_intrinsics_errors = None
                best_mean_pred_intrinsics = None
                seq_results = []
                
                for i in weight_indices:
                    print(f"Evaluating model {i} on SCARED Trajectory {seq}")
                    results = evaluate(
                        opt, 
                        f'{model_dir}/weights_{i}',
                        dataset_name='SCARED',
                        pose_seq=seq,
                        evaluate_pose=True,
                        evaluate_intrinsics=True
                    )
                    
                    # Store this individual result
                    result_row = {
                        'Dataset': 'SCARED', 
                        'Trajectory': seq,
                        'Model': f'weights_{i}'
                    }
                    
                    # Add pose metrics
                    if 'ate_rmse' in results:
                        result_row['ATE_RMSE'] = results['ate_rmse']
                        result_row['ATE_STD'] = results['ate_std']
                        result_row['RPE_Mean'] = results['rpe_mean']
                        result_row['RPE_STD'] = results['rpe_std']
                    
                    # Add intrinsics metrics
                    if 'intrinsics_error' in results:
                        intrinsics_error = results['intrinsics_error']
                        result_row['Scale_Factor'] = intrinsics_error['scale_factor_mean']
                        result_row['FX_Abs_Error'] = intrinsics_error['scale_adjusted_fx_abs_error_mean']
                        result_row['FY_Abs_Error'] = intrinsics_error['scale_adjusted_fy_abs_error_mean']
                        result_row['CX_Abs_Error'] = intrinsics_error['scale_adjusted_cx_abs_error_mean']
                        result_row['CY_Abs_Error'] = intrinsics_error['scale_adjusted_cy_abs_error_mean']
                        result_row['FX_Rel_Error'] = intrinsics_error['scale_adjusted_fx_rel_error_mean']
                        result_row['FY_Rel_Error'] = intrinsics_error['scale_adjusted_fy_rel_error_mean']
                        result_row['Matrix_Norm'] = intrinsics_error['scale_adjusted_matrix_frobenius_norm_mean']
                    
                    seq_results.append(result_row)
                    all_results.append(result_row)
                    
                    if 'ate_rmse' in results and results['ate_rmse'] < min_ate:
                        min_ate = results['ate_rmse']
                        min_re = results['rpe_mean']
                        ate_std = results['ate_std']
                        re_std = results['rpe_std']
                        best_i = i
                        
                        if 'intrinsics_error' in results:
                            best_intrinsics_errors = results['intrinsics_error']
                            best_mean_pred_intrinsics = results['mean_pred_intrinsics']
                
                # Record best result for this sequence
                best_result = {
                    'Dataset': 'SCARED',
                    'Trajectory': seq,
                    'Best_Model': f'weights_{best_i}',
                    'ATE_RMSE': min_ate,
                    'ATE_STD': ate_std,
                    'RPE_Mean': min_re,
                    'RPE_STD': re_std
                }
                
                if best_intrinsics_errors is not None:
                    best_result['Scale_Factor'] = best_intrinsics_errors['scale_factor_mean']
                    best_result['FX_Abs_Error'] = best_intrinsics_errors['scale_adjusted_fx_abs_error_mean']
                    best_result['FY_Abs_Error'] = best_intrinsics_errors['scale_adjusted_fy_abs_error_mean']
                    best_result['CX_Abs_Error'] = best_intrinsics_errors['scale_adjusted_cx_abs_error_mean']
                    best_result['CY_Abs_Error'] = best_intrinsics_errors['scale_adjusted_cy_abs_error_mean']
                    best_result['FX_Rel_Error'] = best_intrinsics_errors['scale_adjusted_fx_rel_error_mean']
                    best_result['FY_Rel_Error'] = best_intrinsics_errors['scale_adjusted_fy_rel_error_mean']
                    best_result['Matrix_Norm'] = best_intrinsics_errors['scale_adjusted_matrix_frobenius_norm_mean']
                
                best_results.append(best_result)
                
                # Write to text file for compatibility
                print(f"SCARED Traj {seq}: Best ATE: {min_ate}±{ate_std}, Best RE: {min_re}±{re_std} @ weights_{best_i}")
                f_result.write(f"SCARED Traj {seq}: Best ATE: {min_ate}±{ate_std}, Best RE: {min_re}±{re_std} @ weights_{best_i}\n")
                
                if best_intrinsics_errors is not None:
                    f_result.write(f"\nBest Intrinsics Results for Trajectory {seq} @ weights_{best_i}:\n")
                    f_result.write(f"Mean Predicted Intrinsics:\n{best_mean_pred_intrinsics}\n\n")
                    f_result.write(f"Mean Scale Factor: {best_intrinsics_errors['scale_factor_mean']:.4f} ± {best_intrinsics_errors['scale_factor_std']:.4f}\n")
                    f_result.write(f"Scale-Adjusted Focal Length X Error: {best_intrinsics_errors['scale_adjusted_fx_abs_error_mean']:.2f} pixels "
                                 f"({best_intrinsics_errors['scale_adjusted_fx_rel_error_mean']:.2f}% relative)\n")
                    f_result.write(f"Scale-Adjusted Focal Length Y Error: {best_intrinsics_errors['scale_adjusted_fy_abs_error_mean']:.2f} pixels "
                                 f"({best_intrinsics_errors['scale_adjusted_fy_rel_error_mean']:.2f}% relative)\n")
                    f_result.write(f"Scale-Adjusted Matrix Frobenius Norm: {best_intrinsics_errors['scale_adjusted_matrix_frobenius_norm_mean']:.2f}\n\n")
                
        elif dataset_name in ['SyntheticColon', 'C3VD', 'test_time_train_1', 'test_time_train_2']:
            # -- SyntheticColon or C3VD --
            ds_dir_dic = {
                'SyntheticColon': 'SyntheticColon_as_SCARED',
                'C3VD': 'C3VD_as_SCARED',
                'test_time_train_1': 'C3VD_test_time_train_1',
                'test_time_train_2': 'C3VD_test_time_train_2'
            }
            test_folders_file = os.path.join(os.path.join(ds_base,ds_dir_dic[dataset_name]), 'splits', 'test_folders.txt')
            
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            
            if debug:
                test_folders = test_folders[:1]  # Only evaluate first test folder in debug mode
            
            min_ate = np.inf
            min_re = np.inf
            ate_std = np.inf
            re_std = np.inf
            best_i = 0
            
            best_intrinsics_errors = None
            best_mean_pred_intrinsics = None
            
            for i in weight_indices:
                ates = []
                res = []
                std_ates = []
                std_res = []
                all_intrinsics_results = []
                
                print(f"Evaluating model {i} on {dataset_name}")
                
                for test_idx, test_folder in enumerate(tqdm(test_folders, desc=f"Evaluating test folders")):
                    full_test_folder = os.path.join(model_dir, test_folder.strip())
                    filename_list_path=os.path.join(full_test_folder, 'traj_test.txt') if os.path.exists(os.path.join(full_test_folder, 'traj_test.txt')) else None
                    results = evaluate(
                        opt, 
                        f'{model_dir}/weights_{i}',
                        dataset_name=dataset_name,
                        pose_seq=test_idx+1,  # Use test index as sequence number
                        evaluate_pose=True,
                        evaluate_intrinsics=True,
                        filename_list_path=filename_list_path,
                    )
                    
                    # Store individual sequence result
                    result_row = {
                        'Dataset': dataset_name, 
                        'Test_Folder': test_folder.strip(),
                        'Model': f'weights_{i}'
                    }
                    
                    # Add pose metrics
                    if 'ate_rmse' in results:
                        result_row['ATE_RMSE'] = results['ate_rmse']
                        result_row['ATE_STD'] = results['ate_std']
                        result_row['RPE_Mean'] = results['rpe_mean']
                        result_row['RPE_STD'] = results['rpe_std']
                        
                        ates.append(results['ate_rmse'])
                        res.append(results['rpe_mean'])
                        std_ates.append(results['ate_std'])
                        std_res.append(results['rpe_std'])
                    
                    # Add intrinsics metrics
                    if 'intrinsics_error' in results:
                        intrinsics_error = results['intrinsics_error']
                        result_row['Scale_Factor'] = intrinsics_error['scale_factor_mean']
                        result_row['FX_Abs_Error'] = intrinsics_error['scale_adjusted_fx_abs_error_mean']
                        result_row['FY_Abs_Error'] = intrinsics_error['scale_adjusted_fy_abs_error_mean']
                        result_row['CX_Abs_Error'] = intrinsics_error['scale_adjusted_cx_abs_error_mean']
                        result_row['CY_Abs_Error'] = intrinsics_error['scale_adjusted_cy_abs_error_mean']
                        result_row['FX_Rel_Error'] = intrinsics_error['scale_adjusted_fx_rel_error_mean']
                        result_row['FY_Rel_Error'] = intrinsics_error['scale_adjusted_fy_rel_error_mean']
                        result_row['Matrix_Norm'] = intrinsics_error['scale_adjusted_matrix_frobenius_norm_mean']
                        all_intrinsics_results.append(results['intrinsics_error'])
                    
                    all_results.append(result_row)
                
                # Calculate average for this model across all test folders
                if len(ates) > 0:
                    avg_result_row = {
                        'Dataset': dataset_name, 
                        'Test_Folder': 'Average',
                        'Model': f'weights_{i}',
                        'ATE_RMSE': np.mean(ates),
                        'ATE_STD': np.mean(std_ates),
                        'RPE_Mean': np.mean(res),
                        'RPE_STD': np.mean(std_res)
                    }
                    
                    # Add average intrinsics metrics
                    if len(all_intrinsics_results) > 0:
                        for key in all_intrinsics_results[0]:
                            if key != 'errors':  # Skip raw error data
                                key_short = key.replace('scale_adjusted_', '').replace('_mean', '')
                                avg_result_row[key_short] = np.mean([result[key] for result in all_intrinsics_results])
                    
                    all_results.append(avg_result_row)
                    
                    # Track best model
                    if np.mean(ates) < min_ate:
                        min_ate = np.mean(ates)
                        min_re = np.mean(res)
                        best_i = i
                        ate_std = np.mean(std_ates)
                        re_std = np.mean(std_res)
                        
                        if len(all_intrinsics_results) > 0:
                            # Compute average of intrinsics results
                            combined_errors = {}
                            for key in all_intrinsics_results[0]:
                                if key != 'errors':  # Skip raw error data
                                    combined_errors[key] = np.mean([result[key] for result in all_intrinsics_results])
                            best_intrinsics_errors = combined_errors
                            # Get mean predicted intrinsics from last result (as an example)
                            if 'mean_pred_intrinsics' in results:
                                best_mean_pred_intrinsics = results['mean_pred_intrinsics']
            
            # Record best result
            best_result = {
                'Dataset': dataset_name,
                'Best_Model': f'weights_{best_i}',
                'ATE_RMSE': min_ate,
                'ATE_STD': ate_std,
                'RPE_Mean': min_re,
                'RPE_STD': re_std
            }
            
            if best_intrinsics_errors is not None:
                best_result['Scale_Factor'] = best_intrinsics_errors['scale_factor_mean']
                best_result['FX_Abs_Error'] = best_intrinsics_errors['scale_adjusted_fx_abs_error_mean']
                best_result['FY_Abs_Error'] = best_intrinsics_errors['scale_adjusted_fy_abs_error_mean']
                best_result['CX_Abs_Error'] = best_intrinsics_errors['scale_adjusted_cx_abs_error_mean']
                best_result['CY_Abs_Error'] = best_intrinsics_errors['scale_adjusted_cy_abs_error_mean']
                best_result['FX_Rel_Error'] = best_intrinsics_errors['scale_adjusted_fx_rel_error_mean']
                best_result['FY_Rel_Error'] = best_intrinsics_errors['scale_adjusted_fy_rel_error_mean']
                best_result['Matrix_Norm'] = best_intrinsics_errors['scale_adjusted_matrix_frobenius_norm_mean']
            
            best_results.append(best_result)
            
            # Write to text file for compatibility
            print(f"{dataset_name}: Best ATE: {min_ate}±{ate_std}, Best RE: {min_re}±{re_std} @ weights_{best_i}")
            f_result.write(f"{dataset_name}: Best ATE: {min_ate}±{ate_std}, Best RE: {min_re}±{re_std} @ weights_{best_i}\n")
            
            if best_intrinsics_errors is not None:
                f_result.write(f"\nBest Intrinsics Results for {dataset_name} @ weights_{best_i}:\n")
                if best_mean_pred_intrinsics is not None:
                    f_result.write(f"Example Mean Predicted Intrinsics:\n{best_mean_pred_intrinsics}\n\n")
                f_result.write(f"Mean Scale Factor: {best_intrinsics_errors['scale_factor_mean']:.4f}\n")
                f_result.write(f"Scale-Adjusted Focal Length X Error: {best_intrinsics_errors['scale_adjusted_fx_abs_error_mean']:.2f} pixels "
                             f"({best_intrinsics_errors['scale_adjusted_fx_rel_error_mean']:.2f}% relative)\n")
                f_result.write(f"Scale-Adjusted Focal Length Y Error: {best_intrinsics_errors['scale_adjusted_fy_abs_error_mean']:.2f} pixels "
                             f"({best_intrinsics_errors['scale_adjusted_fy_rel_error_mean']:.2f}% relative)\n")
                f_result.write(f"Scale-Adjusted Matrix Frobenius Norm: {best_intrinsics_errors['scale_adjusted_matrix_frobenius_norm_mean']:.2f}\n\n")
        
        else:
            print(f"Unknown dataset: {dataset_name}")
            f_result.write(f"Unknown dataset: {dataset_name}\n")
    
    # Write all results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_file, index=False)
        print(f"All results saved to: {csv_file}")
        
        # Write best results to CSV
        if best_results:
            best_df = pd.DataFrame(best_results)
            best_df.to_csv(best_csv_file, index=False)
            print(f"Best results saved to: {best_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pose and intrinsics estimation")
    parser.add_argument('--dataset', type=str, choices=['SCARED', 'SyntheticColon', 
                                                        'C3VD', 'C3VD_testtime_train_1', 
                                                        'C3VD_testtime_train_2'], default='test_time_train_1',
                        help='Dataset to evaluate on (SCARED, SyntheticColon, or C3VD)')
    parser.add_argument('--model_dir', type=str, default='logs/base_model_2/models',
                        help='Directory containing the model weights')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (limited evaluation)')
    parser.add_argument('--pose_only', action='store_true', help='Evaluate only pose, not intrinsics')
    parser.add_argument('--intrinsics_only', action='store_true', help='Evaluate only intrinsics, not pose')
    parser.add_argument('--pose_seq', type=int, default=None, help='Specific pose sequence to evaluate (SCARED only)')
    parser.add_argument('--weights', type=int, default=None, help='Specific weights folder index to evaluate')
    args = parser.parse_args()
    
    # Import options
    try:
        from options_attn_encoder import AttnEncoderOpt
    except ImportError:
        # Fallback - create a basic option class
        class AttnEncoderOpt:
            frame_ids = [0, -1, 1]
            height = 256
            width = 320
    
    if args.pose_only and args.intrinsics_only:
        print("Error: Cannot specify both --pose_only and --intrinsics_only")
        sys.exit(1)
    
    evaluate_pose = not args.intrinsics_only
    evaluate_intrinsics = not args.pose_only
    
    # Single evaluation mode (specific weights and sequence)
    if args.weights is not None and args.pose_seq is not None and args.dataset == 'SCARED':
        print(f"Evaluating SCARED sequence {args.pose_seq} with weights_{args.weights}")
        evaluate(
            AttnEncoderOpt,
            f'{args.model_dir}/weights_{args.weights}',
            dataset_name='SCARED',
            pose_seq=args.pose_seq,
            evaluate_pose=evaluate_pose,
            evaluate_intrinsics=evaluate_intrinsics
        )
    elif args.weights is not None and args.dataset in ['SyntheticColon', 'C3VD', 'test_time_train_1', 'test_time_train_2']:
        print(f"Evaluating {args.dataset} with weights_{args.weights}")
        # Use ds_base defined above

        dataset_path = os.path.join(ds_base, f'{args.dataset}_as_SCARED')
        test_folders_file = os.path.join(dataset_path, 'splits', 'test_folders.txt')
        with open(test_folders_file, 'r') as f:
            test_folders = f.readlines()[:1]  # Just evaluate first test folder
        test_folder = test_folders[0].strip()
        full_test_folder = os.path.join(dataset_path, test_folder)
        filename_list_path=os.path.join(full_test_folder, 'traj_test.txt') if os.path.exists(os.path.join(full_test_folder, 'traj_test.txt')) else None
        evaluate(
            AttnEncoderOpt,
            f'{args.model_dir}/weights_{args.weights}',
            dataset_name=args.dataset,
            evaluate_pose=evaluate_pose,
            evaluate_intrinsics=evaluate_intrinsics,
            filename_list_path=filename_list_path,
        )
    else:
        # Batch evaluation mode
        evaluate_batch(
            AttnEncoderOpt,
            args.dataset,
            args.model_dir,
            debug=args.debug
        )
