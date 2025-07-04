#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified depth evaluation script for SCARED dataset based on trainer configuration.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# Add paths for imports
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

from dataset import SCAREDRAWDataset
from utils import readlines

# Use the same dataset path as trainer
DATASET_BASE = '/mnt/c/Users/14152/ZCH/Dev/datasets'

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate(test_dataloader, depth_model, opt, min_depth=0.1, max_depth=100.0):
    """
    Simplified depth evaluation.
    
    Args:
        test_dataloader: DataLoader for test dataset
        depth_model: Trained depth model
        opt: Configuration options
        min_depth: Minimum depth for evaluation
        max_depth: Maximum depth for evaluation
        
    Returns:
        Dictionary with depth evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_model.to(device)
    depth_model.eval()
    
    pred_depths = []
    gt_depths = []
    
    print("Running depth evaluation...")
    
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Processing"):
            input_color = data[("color", 0, 0)].to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Get depth prediction
            output = depth_model(input_color)
            pred_depth = output[("depth", 0, 0)]
            
            # Get ground truth depth from npz files
            if ("depth_gt", 0, 0) in data:
                gt_depth = data[("depth_gt", 0, 0)]
                
                pred_depth = pred_depth.cpu()[:, 0].numpy()
                gt_depth = gt_depth.cpu()[:, 0].numpy()
                
                # Process each sample in the batch
                for i in range(pred_depth.shape[0]):
                    gt = gt_depth[i]
                    pred = pred_depth[i]
                    
                    # Apply depth range mask
                    mask = np.logical_and(gt > min_depth, gt < max_depth)
                    
                    if mask.sum() > 0:
                        gt_depths.append(gt[mask])
                        pred_depths.append(pred[mask])
    
    if len(gt_depths) == 0:
        print("Warning: No valid depth data found for evaluation")
        return {}
    
    # Concatenate all depths
    gt_depths = np.concatenate(gt_depths)
    pred_depths = np.concatenate(pred_depths)
    
    # Compute metrics
    errors = compute_errors(gt_depths, pred_depths)
    
    print("Depth evaluation results:")
    print(f"abs_rel: {errors[0]:.4f}")
    print(f"sq_rel: {errors[1]:.4f}")
    print(f"rmse: {errors[2]:.4f}")
    print(f"rmse_log: {errors[3]:.4f}")
    print(f"a1: {errors[4]:.4f}")
    print(f"a2: {errors[5]:.4f}")
    print(f"a3: {errors[6]:.4f}")
    
    return {
        'abs_rel': errors[0],
        'sq_rel': errors[1], 
        'rmse': errors[2],
        'rmse_log': errors[3],
        'a1': errors[4],
        'a2': errors[5],
        'a3': errors[6]
    }

def evaluate_scared_depth(dataset_path, filenames, depth_model, opt):
    """
    Evaluate depth on SCARED dataset.
    
    Args:
        dataset_path: Path to SCARED dataset
        filenames: List of test filenames
        depth_model: Trained depth model
        opt: Configuration options
        
    Returns:
        Dictionary with evaluation results
    """
    # Create dataset
    dataset = SCAREDRAWDataset(
        dataset_path, filenames,
        opt.height, opt.width,
        [0], 1, is_train=False,
        load_depth_from_npz=True  # Load GT depth
    )
    
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.num_workers,
        pin_memory=True, drop_last=False
    )
    
    return evaluate(dataloader, depth_model, opt)

# Simplified interface for compatibility
def evaluate_depth_scared(dataset_path, filenames, depth_model, opt):
    """Simplified interface for SCARED depth evaluation."""
    return evaluate_scared_depth(dataset_path, filenames, depth_model, opt)
