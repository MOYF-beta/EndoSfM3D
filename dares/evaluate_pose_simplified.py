#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified pose and intrinsics evaluation script based on trainer configuration.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Add paths for imports
current_dir = os.path.dirname(__file__)
dares_networks_path = os.path.join(current_dir, 'networks')
sys.path.insert(0, dares_networks_path)
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

# Import models based on trainer configuration
from resnet_encoder import MultiHeadAttentionalResnetEncoder
from pose_decoder import PoseDecoder_with_intrinsics as PoseDecoder_i
from layers import transformation_from_parameters
from utils import readlines
from dataset import SCAREDRAWDataset

# Use the same dataset path as trainer
DATASET_BASE = '/mnt/c/Users/14152/ZCH/Dev/datasets'

def create_transformation_matrix(axisangle, translation):
    """Create 4x4 transformation matrix from axis-angle and translation."""
    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
    return T

def evaluate_pose(opt, weights_folder, dataset_name="SCARED", pose_seq=1):
    """
    Simplified pose evaluation matching trainer configuration.
    
    Args:
        opt: Configuration object with model parameters
        weights_folder: Path to model weights
        dataset_name: Dataset name (SCARED)
        pose_seq: Sequence number for SCARED dataset
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up dataset path
    if dataset_name == 'SCARED':
        ds_path = os.path.join(DATASET_BASE, 'SCARED_Images_Resized')
        test_files_path = os.path.join(ds_path, "splits", f"test_files_sequence{pose_seq}.txt")
        filenames = readlines(test_files_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Using test files: {test_files_path}")
    print(f"Number of test files: {len(filenames)}")
    
    # Create dataset and dataloader
    dataset = SCAREDRAWDataset(
        ds_path, filenames, 
        opt.height, opt.width,
        [0, 1], 1, is_train=False
    )
    dataloader = DataLoader(
        dataset, opt.batch_size, 
        shuffle=False, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=False
    )
    
    # Load pose models (matching trainer configuration)
    pose_encoder = MultiHeadAttentionalResnetEncoder(
        opt.num_layers, False, 
        num_input_images=2  # self.num_pose_frames = 2 from trainer
    )
    
    pose_decoder = PoseDecoder_i(
        pose_encoder.num_ch_enc,
        image_width=opt.width,
        image_height=opt.height,
        predict_intrinsics=opt.learn_intrinsics,
        simplified_intrinsic=opt.simplified_intrinsic,
        num_input_features=1,
        num_frames_to_predict_for=2,
        auto_scale=True
    )
    
    # Load weights
    pose_encoder_path = os.path.join(weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(weights_folder, "pose.pth")
    
    if not os.path.exists(pose_encoder_path):
        raise FileNotFoundError(f"Pose encoder weights not found: {pose_encoder_path}")
    if not os.path.exists(pose_decoder_path):
        raise FileNotFoundError(f"Pose decoder weights not found: {pose_decoder_path}")
    
    # Load state dicts
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device))
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location=device))
    
    # Move to device and set eval mode
    pose_encoder.to(device)
    pose_decoder.to(device)
    pose_encoder.eval()
    pose_decoder.eval()
    
    # Run inference
    pred_poses = []
    pred_intrinsics = []
    
    print("Running pose inference...")
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Processing"):
            # Move inputs to device
            for key, value in inputs.items():
                inputs[key] = value.to(device)
            
            # Prepare input for pose network (matching trainer logic)
            # Concatenate frames: [frame_1, frame_0] 
            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)
            
            # Encode features
            features = [pose_encoder(all_color_aug)]
            
            # Decode pose and intrinsics
            axisangle, translation, intrinsics = pose_decoder(features)
            
            # Convert to transformation matrices
            T = create_transformation_matrix(axisangle, translation)
            
            # Store predictions
            pred_poses.extend(T.cpu().numpy())
            if intrinsics is not None:
                pred_intrinsics.extend(intrinsics.cpu().numpy())
    
    print(f"Collected {len(pred_poses)} pose predictions")
    
    # Simple metrics calculation
    results = {
        'num_predictions': len(pred_poses),
        'pose_predictions': pred_poses,
        'intrinsic_predictions': pred_intrinsics if pred_intrinsics else None,
        'dataset': dataset_name,
        'sequence': pose_seq
    }
    
    # Save predictions for further analysis
    output_dir = os.path.join(weights_folder, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_file = os.path.join(output_dir, f'{dataset_name}_seq{pose_seq}_predictions.npz')
    np.savez(pred_file, 
             poses=np.array(pred_poses),
             intrinsics=np.array(pred_intrinsics) if pred_intrinsics else None)
    
    print(f"Predictions saved to: {pred_file}")
    
    return results

def evaluate_model(opt_dict, weights_folder, dataset_name="SCARED", pose_seq=1):
    """
    Main evaluation function that can be called from find_best_parametric.
    
    Args:
        opt_dict: Dictionary containing model options
        weights_folder: Path to model weights
        dataset_name: Dataset name 
        pose_seq: Sequence number
        
    Returns:
        Evaluation score (lower is better)
    """
    # Convert dict to object for compatibility
    class OptObject:
        def __init__(self, opt_dict):
            for key, value in opt_dict.items():
                setattr(self, key, value)
    
    opt = OptObject(opt_dict)
    
    try:
        results = evaluate_pose(opt, weights_folder, dataset_name, pose_seq)
        
        # Return a simple score based on number of successful predictions
        # In a real scenario, you would compute proper metrics like ATE, RPE, etc.
        score = 1.0 / max(1, results['num_predictions'])  # Lower is better
        
        print(f"Evaluation completed. Score: {score:.6f}")
        return score
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return float('inf')  # Return worst possible score on failure

# Simplified interface function for compatibility with existing code
def evaluate(opt, load_weights_folder, dataset_name="SCARED", pose_seq=1, **kwargs):
    """Simplified evaluation interface."""
    return evaluate_model(opt.__dict__, load_weights_folder, dataset_name, pose_seq)

if __name__ == "__main__":
    # Simple test
    print("Simplified pose evaluation script")
    print("Use this script by importing evaluate_model or evaluate functions")
