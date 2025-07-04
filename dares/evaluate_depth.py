from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
try:
    from layers import disp_to_depth 
except ImportError:
    from .layers import disp_to_depth 
try:
    from options import MonodepthOptions
except ImportError:
    try:
        from ..options import MonodepthOptions
    except ImportError:
        # Create a placeholder class if options is not available
        class MonodepthOptions:
            pass

# Default splits directory - can be overridden
platform = 'local' if os.getcwd().startswith('/mnt') else 'cluster'
ds_base = '/mnt/c/Users/14152/ZCH/Dev/datasets' if platform == 'local' else '/cluster/project7/Llava_2024/changhao/datasets'
splits_dir = os.path.join(ds_base, 'C3VD_as_SCARED', 'splits')
#cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths using torch on GPU
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(), a1.item(), a2.item(), a3.item()


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


from torchvision.utils import save_image

def evaluate(opt, ds_and_model = {}, frames_input = [0], load_depth_from_npz = False, 
             show_images = True, auto_scale = True,image_save_countdown = 20, output_dir = None):
    """Evaluates a pretrained model using a specified test set
    """
    dataloader = ds_and_model['dataloader']
    depth_model = ds_and_model['depth_model']
    
    errors = []
    ratios = []
    image_index = 0  # Initialize image index

    # Make sure the output directory exists
    if output_dir is None:
        output_dir = './eval_images/'
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        num_input_frames = len(frames_input)
        for data in dataloader:
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
            MIN_DEPTH = 1e-2
            MAX_DEPTH = 150
            pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp[:, 0]

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            gt_depth = data["depth_gt"].cuda()
            gt_height, gt_width = gt_depth.shape[-2:]

            pred_disp = torch.stack([torch.nn.functional.interpolate(d.unsqueeze(0).unsqueeze(0), size=(gt_height, gt_width), mode='bilinear', align_corners=False).squeeze() for d in pred_disp])
            pred_depth = 1 / pred_disp

            # Save images every 1000 frames
            if show_images and image_index % image_save_countdown == 0:
                # Use a non-interactive backend for matplotlib to avoid Tkinter errors
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                # Save the input color image
                save_image(input_color[0], os.path.join(output_dir, f"{image_index}_color.png"))

                # Save the ground truth depth image
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(gt_depth[0].squeeze().cpu().numpy(), cmap='plasma')
                plt.savefig(os.path.join(output_dir, f"{image_index}_gt.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                # Save the predicted disparity image
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(pred_depth[0].squeeze().cpu().numpy(), cmap='plasma')
                plt.savefig(os.path.join(output_dir, f"{image_index}_pred.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            gt_depth = gt_depth[mask]
            if len(mask.shape) == 4:
                mask = mask.squeeze(1)
            pred_depth = pred_depth[mask]

            
            if auto_scale:
                # Automatically compute the optimal scaling factor using mean scaling
                scale = torch.mean(1/gt_depth[gt_depth > 0]) / torch.mean(1/pred_depth[gt_depth > 0])
                pred_depth /= scale
            else:
                pred_depth *= opt.pred_depth_scale_factor
                if not opt.disable_median_scaling:
                    ratio = torch.median(gt_depth) / torch.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio

            pred_depth = torch.clamp(pred_depth, min=MIN_DEPTH, max=MAX_DEPTH)

            

            errors.append(compute_errors(gt_depth, pred_depth))
            image_index += 1  # Increment image index for naming

    if not opt.disable_median_scaling:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)

    mean_errors = torch.tensor(errors).mean(0)

    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\", end='')

    return mean_errors

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
