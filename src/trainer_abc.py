import json
import os
import random
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from dataset import DataLoaderX as DataLoader
from torch.utils.data import RandomSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

# Setup paths for direct imports
import os
import sys
current_dir = os.path.dirname(__file__)
dares_path = os.path.join(current_dir, '..', 'dares')
endo3dac_utils_path = os.path.join(current_dir, '..', 'endo3dac', 'utils')
sys.path.insert(0, dares_path)
sys.path.insert(0, endo3dac_utils_path)

# Import from reorganized packages
from layers import *
from utils import *
from endodac.utils.layers import SSILoss  # This is from endo3dac.utils.layers

import torch.nn as nn
import numpy as np

def undistort_image(image, K, K_distortion, P_distortion):
    """
    Apply undistortion to image using camera intrinsics and distortion coefficients.
    
    Args:
        image: Input image tensor [B, C, H, W]
        K: Camera intrinsics matrix [B, 4, 4] or [B, 3, 3]
        K_distortion: Radial distortion coefficients [B, 3] (k1, k2, k3)
        P_distortion: Tangential distortion coefficients [B, 2] (p1, p2)
    
    Returns:
        undistorted_image: Undistorted image tensor [B, C, H, W]
        K_undistorted: Updated intrinsics matrix for undistorted image
    """
    B, C, H, W = image.shape
    device = image.device
    
    # Extract intrinsics parameters
    if K.shape[-1] == 4:
        fx = K[:, 0, 0]  # [B]
        fy = K[:, 1, 1]  # [B]
        cx = K[:, 0, 2]  # [B]
        cy = K[:, 1, 2]  # [B]
    else:  # 3x3 matrix
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
      # Extract distortion coefficients
    if K_distortion.dim() == 1:
        # If distortion coefficients are 1D (batch mean), expand to batch size
        K_distortion = K_distortion.unsqueeze(0).expand(B, -1)
    if P_distortion.dim() == 1:
        P_distortion = P_distortion.unsqueeze(0).expand(B, -1)
        
    k1 = K_distortion[:, 0]  # [B]
    k2 = K_distortion[:, 1]  # [B]
    k3 = K_distortion[:, 2]  # [B]
    p1 = P_distortion[:, 0]  # [B]
    p2 = P_distortion[:, 1]  # [B]
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    undistorted_images = []
    updated_K_matrices = []
    
    for b in range(B):
        # Normalize coordinates to [-1, 1] range relative to principal point
        x_norm = (x_coords - cx[b]) / fx[b]
        y_norm = (y_coords - cy[b]) / fy[b]
        
        # Calculate r^2
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2**3
        
        # Apply radial distortion
        radial_factor = 1 + k1[b] * r2 + k2[b] * r4 + k3[b] * r6
        
        # Apply tangential distortion
        tangential_x = 2 * p1[b] * x_norm * y_norm + p2[b] * (r2 + 2 * x_norm**2)
        tangential_y = p1[b] * (r2 + 2 * y_norm**2) + 2 * p2[b] * x_norm * y_norm
        
        # Compute distorted coordinates
        x_distorted = x_norm * radial_factor + tangential_x
        y_distorted = y_norm * radial_factor + tangential_y
        
        # Convert back to pixel coordinates
        x_distorted_px = x_distorted * fx[b] + cx[b]
        y_distorted_px = y_distorted * fy[b] + cy[b]
        
        # Normalize coordinates for grid_sample (to [-1, 1] range)
        grid_x = 2.0 * x_distorted_px / (W - 1) - 1.0
        grid_y = 2.0 * y_distorted_px / (H - 1) - 1.0
        
        # Create sampling grid [H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        
        # Apply undistortion by sampling from distorted locations
        undistorted_img = F.grid_sample(
            image[b:b+1], grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        undistorted_images.append(undistorted_img)
        
        # For undistorted image, we can use the original intrinsics
        # as the distortion has been removed
        if K.shape[-1] == 4:
            K_undist = K[b:b+1].clone()
        else:
            K_undist = torch.eye(4, device=device).unsqueeze(0)
            K_undist[0, :3, :3] = K[b]
        updated_K_matrices.append(K_undist)
    
    undistorted_image = torch.cat(undistorted_images, dim=0)
    K_undistorted = torch.cat(updated_K_matrices, dim=0)
    
    return undistorted_image, K_undistorted

class GlobalRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

class ConcatDatasetSampler(Sampler):
    def __init__(self, dataset):
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            self.concat_dataset = dataset
            self.cumulative_sizes = dataset.cumulative_sizes
            print(f'cumulative_sizes: {self.cumulative_sizes}')
        else:
            self.concat_dataset = None
            self.dataset = dataset
            print("\033[91m WARNING: Dataset is not a ConcatDataset, using GlobalRandomSampler instead.\033[0m")

    def __iter__(self):
        if self.concat_dataset:
            indices = []
            start = 0
            for end in self.cumulative_sizes:
                dataset_len = end - start
                perm = torch.randperm(dataset_len).tolist()
                global_perm = [p + start for p in perm]
                indices.extend(global_perm)
                start = end
            return iter(indices)
        else:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            return iter(indices)

    def __len__(self):
        if self.concat_dataset:
            return len(self.concat_dataset)
        else:
            return len(self.dataset)
    
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from abc import ABC, abstractmethod
class Trainer(ABC):

    @abstractmethod
    def load_model(self):
        pass

    def __init__(self, model_name, log_dir, options, train_eval_ds={},
                  pretrained_root_dir=None, merge_val_as_train=False, 
                  use_supervised_loss = False, debug = False, use_af_pose=False,
                  accumulate_position_gradients_for_pose=False, save_samples_num=10):
        if debug:
            print("\033[91m WARNING: Debug mode activated, only train 1 epoch\033[0m")
            options.num_epochs = 2
            options.batch_size = 8
        if options.self_ssi:
            assert options.self_ssi_constraint is not None, "self_ssi_constraint must be set if self_ssi is enabled"
        self.use_af_pose = use_af_pose
        self.accumulate_position_gradients_for_pose = accumulate_position_gradients_for_pose
        self.opt = options
        self.log_path = os.path.join(log_dir, model_name)
        self.use_supervised_loss = use_supervised_loss
        self.models = {}  
        self.param_monodepth = []  
        self.param_pose_net = []
        self.save_samples_num = save_samples_num
        
        self.device = device
        self.train_pos = False
        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        self.pretrained_root_dir = pretrained_root_dir
        self.load_model()

        # Initialize optimizer and scheduler
        self.optimizer_pose = optim.Adam(self.param_pose_net, self.opt.pos_learning_rate, eps=1e-7, weight_decay=self.opt.weight_decay)
        self.lr_scheduler_pose = optim.lr_scheduler.StepLR(self.optimizer_pose, self.opt.scheduler_step_size, 0.1)
        
        self.optimizer_depth = optim.Adam(self.param_monodepth, self.opt.learning_rate, eps=1e-7, weight_decay=self.opt.weight_decay_pose)
        self.lr_scheduler_depth = optim.lr_scheduler.StepLR(self.optimizer_depth, self.opt.scheduler_step_size, 0.1)


        train_dataset = train_eval_ds["train"]
        val_dataset = train_eval_ds["val"]
        
        if 'train_sampler' in train_eval_ds:
            train_sampler = train_eval_ds['train_sampler'](train_dataset)
        else:
            train_sampler = GlobalRandomSampler(train_dataset)
        if merge_val_as_train:
            print("\033[91m WARNING: Merging validation dataset into training dataset\033[0m")
            combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            self.train_loader = DataLoader(
            combined_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
            self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        
        num_train_samples = len(train_eval_ds['train'])
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            self.ms_ssim.to(self.device)
        # Add Self-SSI loss initialization
        if self.opt.self_ssi:
            self.self_ssi = SSILoss()
            self.self_ssi.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
    
    def set_train(self, train_position_only=False, freeze_depth=False):
        """Enable or disable gradients based on optimizer groups."""
        # Switch model modes: when training position only, keep depth_model in eval, others in train; otherwise all train
        for name, model in self.models.items():
            model.train()
        # Freeze/unfreeze depth parameters
        for group in self.optimizer_depth.param_groups:
            for p in group['params']:
                p.requires_grad = not train_position_only
        # Freeze/unfreeze pose parameters
        for group in self.optimizer_pose.param_groups:
            for p in group['params']:
                p.requires_grad = train_position_only  or self.accumulate_position_gradients_for_pose
        if freeze_depth:
            for param in self.models["depth_model"].parameters():
                param.requires_grad = False
            self.models["depth_model"].eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        self.models["pose_encoder"].eval() if "pose_encoder" in self.models else None
        self.models["pose"].eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        
        # Save samples after training if save_samples_num > 0
        if self.save_samples_num > 0:
            self.save_depth_samples()

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        print("-"*20 + "Training"+ "-"*20)
        summed_loss = 0
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            
            # Train position
            self.set_train(train_position_only=True)
            outputs, losses = self.process_batch_position(inputs)
            loss_pose = losses["loss"]
            if self.accumulate_position_gradients_for_pose:
                for param in self.param_pose_net:
                    if param.grad is not None:
                        param.grad.data *= 0.1
                loss_pose.backward()
                self.optimizer_pose.step()
                self.optimizer_pose.zero_grad()    
            else:
                self.optimizer_pose.zero_grad()
                loss_pose.backward()
                self.optimizer_pose.step()            # Train depth
            ssi_warming_up = self.opt.self_ssi and self.epoch < int(self.opt.warm_up_step)
            # print(f"Epoch {self.epoch}, batch {batch_idx}, ssi_warming_up: {ssi_warming_up}",end = '\r')
            random_train = random.random() < 0.1 * (self.epoch+1)
            
            # Determine whether to train depth network
            # - If self_ssi is disabled: always train
            # - If self_ssi is enabled but not in warming up: always train  
            # - If self_ssi is enabled and in warming up: train with probability
            should_train_depth = True # not self.opt.self_ssi or not ssi_warming_up or random_train
            
            if should_train_depth:
                self.set_train(train_position_only=False, freeze_depth=ssi_warming_up if self.opt.self_ssi else False)
                outputs, losses = self.process_batch_depth(inputs)
                loss_depth = losses["loss"]

                self.optimizer_depth.zero_grad()
                loss_depth.backward()
                self.optimizer_depth.step()
            else:
                self.set_eval()
                outputs, losses = self.process_batch_depth(inputs)
                loss_depth = losses["loss"]

            duration = time.time() - before_op_time

            # Log training
            phase = batch_idx % self.opt.log_frequency == 0
            summed_loss += loss_depth.item()
            if phase:
                avg_loss = loss_depth.item() if batch_idx == 0 else summed_loss/self.opt.log_frequency

                self.log_time(batch_idx, duration, avg_loss)
                self.log("train", outputs, losses)
                summed_loss = 0

            self.step += 1

        self.lr_scheduler_pose.step()
        self.lr_scheduler_depth.step()

    def process_batch_position(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = {}
        outputs=self.predict_poses(inputs, outputs)
        
        losses = self.compute_position_losses(inputs, outputs)

        return outputs, losses

    @abstractmethod
    def get_depth_input(self, inputs):
        multiframe_input = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)
        batch_size, num_channels, height, width = multiframe_input.shape
        multiframe_input = multiframe_input.view(batch_size, self.num_input_frames, num_channels // self.num_input_frames, height, width).permute(0, 2, 1, 3, 4)
        

    def process_batch_depth(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = {}
        depth_input = self.get_depth_input(inputs)
        outputs = self.models["depth_model"](depth_input)
        outputs.update(self.predict_poses(inputs, outputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_depth_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, disps=None):
        """Predict poses and other outputs for monocular sequences."""
        outputs = {}

        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":

                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # Position
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        # Forward position
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("registration", scale, f_i)] = self.spatial_transform(
                            inputs[("color", f_i, 0)],
                            outputs[("position", "high", scale, f_i)]
                        )

                        # Reverse position
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                            self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)]
                        )

                    # Transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("refined", scale, f_i)] = (
                            outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()
                            + inputs[("color", 0, 0)]
                        )
                        outputs[("refined", scale, f_i)] = torch.clamp(
                            outputs[("refined", scale, f_i)], min=0.0, max=1.0
                        )

                    # Pose (if disps is provided)
                    if disps is not None:
                        # NOTE test: merge appearance flow onto pose
                        if self.use_af_pose:
                            inputs_all = [pose_feats[f_i], pose_feats[0], outputs[("transform", "high", 0, f_i)]]
                        else:
                            inputs_all = [pose_feats[f_i], pose_feats[0]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                        if self.opt.learn_intrinsics:
                            axisangle, translation, intrinsics = self.models["pose"](pose_inputs)
                            outputs["estimated_intrinsics"] = intrinsics
                        else:
                            axisangle, translation = self.models["pose"](pose_inputs)

                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0]
                        )
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
            
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)                
                if self.opt.predict_distortion:
                    K, K_distortion, P_distortion = outputs["estimated_intrinsics"]
                    
                    # Apply undistortion to the target frame image
                    target_image = inputs[("color", frame_id, source_scale)]
                    undistorted_target, K_undistorted = undistort_image(
                        target_image, K, K_distortion, P_distortion)
                    
                    # Use undistorted intrinsics for projection
                    inv_K_undistorted = torch.inverse(K_undistorted)
                    cam_points = self.backproject_depth[source_scale](
                        depth, inv_K_undistorted)
                    pix_coords = self.project_3d[source_scale](
                        cam_points, K_undistorted, T)
                      # Sample from undistorted image
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        undistorted_target,
                        pix_coords,
                        padding_mode="border",
                        align_corners=True)
                    
                elif self.opt.learn_intrinsics:
                    K = outputs["estimated_intrinsics"]
                    inv_K = torch.inverse(K)

                    cam_points = self.backproject_depth[source_scale](
                        depth, inv_K)
                    pix_coords = self.project_3d[source_scale](
                        cam_points, K, T)
                else:
                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # Only apply grid_sample for non-distortion cases
                # (distortion case already handled above)
                if not self.opt.predict_distortion:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border",
                        align_corners=True)

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                # Add generation of transformed disparity for self-ssi
                if self.opt.self_ssi and self.epoch >= self.opt.warm_up_step:
                    t_to_s_disp = F.grid_sample(
                        outputs[("disp", scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border",
                        align_corners=True)
                    outputs[("t_to_s_disp", frame_id, scale)] = t_to_s_disp
       

    def compute_supervised_loss(self, inputs, outputs, debug=False):
        if 'depth_gt' in inputs.keys() and inputs['depth_gt'].any():
            depth_gt = inputs['depth_gt']
        else:
            return 0
        depth_pred = outputs[("depth", 0, 0)] * self.opt.max_depth
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
            
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        return nn.L1Loss()(depth_pred, depth_gt)
        

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            reprojection_loss = 0.9 * ms_ssim_loss + 0.1 * l1_loss

        return reprojection_loss
    
    def compute_of_loss(self, of, depth, pose, mask=None):
        '''with depth and pose, we are able to synthesize the optical flow
        compute the loss between the synthesized optical flow and the predicted optical flow
        '''
        pass

    def compute_position_losses(self, inputs, outputs):
        """Compute losses for position optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += get_smooth_loss(outputs[("position", scale, frame_id)], color)
                loss_registration += (
                    self.compute_reprojection_loss(
                        outputs[("registration", scale, frame_id)], 
                        outputs[("refined", scale, frame_id)]#.detach() # important : now dont use transform loss, instead, this is combined into the registration loss
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs):
        """Compute losses for depth optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            # Add self-ssi accumulation
            loss_self_ssi = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                # Compute reprojection loss
                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)], 
                        outputs[("refined", scale, frame_id)]
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()                # Add self-ssi loss term
                if self.opt.self_ssi and self.epoch >= self.opt.warm_up_step:
                    # Ensure both tensors have the same size for SSI loss
                    disp_original = outputs[("disp", scale)]
                    disp_transformed = outputs[("t_to_s_disp", frame_id, scale)]
                    
                    # Resize transformed disparity to match original if needed
                    if disp_original.shape != disp_transformed.shape:
                        disp_transformed = F.interpolate(
                            disp_transformed, 
                            size=disp_original.shape[-2:], 
                            mode="bilinear", 
                            align_corners=True
                        )
                    
                    # Also ensure mask has the same size
                    mask = occu_mask_backward
                    if mask.shape != disp_original.shape:
                        mask = F.interpolate(
                            mask.float(), 
                            size=disp_original.shape[-2:], 
                            mode="bilinear", 
                            align_corners=True
                        )
                    
                    loss_self_ssi += self.self_ssi(
                        disp_original,
                        disp_transformed,
                        mask)

            # Add weighted self-ssi constraint
            if self.opt.self_ssi and self.epoch >= self.opt.warm_up_step:
                loss += self.opt.self_ssi_constraint * loss_self_ssi

            # Compute disparity smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            disp_smooth_loss = get_smooth_loss(norm_disp, color)

            # Combine losses
            loss += loss_reprojection / 2.0
            loss += self.opt.disparity_smoothness * disp_smooth_loss / (2 ** scale)

            # supervised loss
            if self.use_supervised_loss:
                loss += self.compute_supervised_loss(inputs, outputs)  * 0.001
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss depth: {:.5f} | time elapsed: {} | time left: {} | lr_pos: {} | lr_depth: {}"
        
        lr_pos = str(self.optimizer_pose.param_groups[0]['lr'])
        lr_depth = str(self.optimizer_depth.param_groups[0]['lr'])
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)
                                  ,lr_pos, lr_depth))

    def log(self, mode, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
            if ("disp", 0) in outputs:
                for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
                    writer.add_image(
                        "disp_0",
                        normalize_image(outputs[("disp", 0)][j]), self.step)
                    writer.add_image(
                        "refined_0",
                        normalize_image(outputs[("refined", 0, 1)][j]), self.step)
                    writer.add_image(
                        "registration_0",
                        normalize_image(outputs[("registration", 0, 1)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer_pose.state_dict(), save_path)

    def save_depth_samples(self):
        """Save depth visualization samples from training dataset"""
        print(f"\033[92mSaving {self.save_samples_num} depth visualization samples...\033[0m")
        
        # Create samples directory
        samples_dir = os.path.join(self.log_path, "depth_samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        
        # Set model to evaluation mode
        self.set_eval()
        
        # Get training dataset
        train_dataset = self.train_loader.dataset
        if hasattr(train_dataset, 'datasets'):  # ConcatDataset
            dataset_len = len(train_dataset.datasets[0])
        else:
            dataset_len = len(train_dataset)
        
        # Calculate indices for evenly spaced samples
        step = max(1, dataset_len // self.save_samples_num)
        sample_indices = [i * step for i in range(self.save_samples_num)]
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(sample_indices):
                if sample_idx >= dataset_len:
                    break
                    
                # Get sample from dataset
                if hasattr(train_dataset, 'datasets'):  # ConcatDataset
                    sample = train_dataset.datasets[0][sample_idx]
                else:
                    sample = train_dataset[sample_idx]
                
                # Prepare inputs
                inputs = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.unsqueeze(0).to(self.device)
                    else:
                        inputs[key] = value
                  # Get depth prediction
                try:
                    depth_input = self.get_depth_input(inputs)
                    outputs = self.models["depth_model"](depth_input)
                except Exception as e:
                    print(f"Warning: Failed to process sample {sample_idx}: {e}")
                    continue
                
                # Convert disparity to depth
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                
                # Save original image
                color_img = inputs[("color", 0, 0)].squeeze().cpu()
                if color_img.shape[0] == 3:  # RGB
                    color_img = color_img.permute(1, 2, 0)
                color_img = (color_img.numpy() * 255).astype(np.uint8)
                
                color_pil = Image.fromarray(color_img)
                color_path = os.path.join(samples_dir, f"sample_{idx:03d}_color.png")
                color_pil.save(color_path)
                
                # Save depth visualization
                depth_img = depth.squeeze().cpu().numpy()
                plt.figure(figsize=(10, 6))
                plt.imshow(depth_img, cmap='plasma')
                plt.colorbar(label='Depth (m)')
                plt.title(f'Predicted Depth - Sample {idx}')
                plt.axis('off')
                
                depth_path = os.path.join(samples_dir, f"sample_{idx:03d}_depth.png")
                plt.savefig(depth_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"Saved sample {idx+1}/{self.save_samples_num}: {color_path}, {depth_path}")
        
        print(f"\033[92mCompleted saving depth samples to: {samples_dir}\033[0m")