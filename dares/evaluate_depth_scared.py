from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

try:
    from layers import disp_to_depth
except ImportError:
    from .layers import disp_to_depth
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
import datasets
import networks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cv2.setNumThreads(0)

# Constants that were imported from exp_setup_local
platform = 'local' if os.getcwd().startswith('/mnt') else 'cluster'
ds_base = '/workspace/data'
splits_dir = os.path.join(ds_base, 'SCARED_Images_Resized', 'splits')
gt_path = os.path.join(splits_dir, "gt_depths.npz")
if os.path.exists(gt_path):
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
else:
    gt_depths = None
    print(f"Warning: GT depths not found at {gt_path}")
STEREO_SCALE_FACTOR = 5.4

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # Remove nan values
    valid_mask = np.isfinite(gt) & np.isfinite(pred)
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp



def evaluate(opt, ds_and_model = {}, frames_input = [0], load_depth_from_npz = False, 
             save_images = True, auto_scale = True, image_save_countdown = 5, output_dir = './eval_images/'):
    """Evaluates a pretrained model using a specified test set
    """
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
    global gt_depths
    MIN_DEPTH = 1e-2
    MAX_DEPTH = 150

    dataloader = ds_and_model['dataloader']
    depth_model = ds_and_model['depth_model']
    pred_disps = []
    errors = []
    ratios = []

    with torch.no_grad():
        num_input_frames = len(frames_input)
        for i,data in enumerate(dataloader):
            if num_input_frames == 1:
                input_color = data[("color", frames_input[0], 0)].cuda()
            else:
                input_color = torch.cat([data[("color_aug", i, 0)] for i in frames_input], 1)
                batch_size, num_channels, height, width = input_color.shape
                input_color = input_color.view(batch_size, num_input_frames, num_channels // num_input_frames, height, width).permute(0, 2, 1, 3, 4)
                input_color = input_color.cuda()
            
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_model(input_color)
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            if save_images and i % image_save_countdown == 0:


                plt.imsave(os.path.join(output_dir, f"{i}_color.png"), input_color[0].permute(1, 2, 0).cpu().numpy())

                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(gt_depths[i*16], cmap='plasma')
                plt.savefig(os.path.join(output_dir, f"{i}_gt.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                # Save the predicted depth image
                pred_depth = 1/pred_disp
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(pred_depth[0], cmap='plasma')
                plt.savefig(os.path.join(output_dir, f"{i}_pred.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)


    pred_disps = np.concatenate(pred_disps)

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        image_index = 0
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1/pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if auto_scale:
            # Automatically compute the optimal scaling factor using least squares
            scale = np.mean(1/gt_depth[gt_depth > 0]) / np.mean(1/pred_depth[gt_depth > 0])
            pred_depth /= scale
        else:
            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)

    mean_errors = np.array(errors).mean(0)

    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\", end='')

    return mean_errors
