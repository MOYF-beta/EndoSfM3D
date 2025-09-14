import json
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from layers import *
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from abc import ABC, abstractmethod
class Trainer(ABC):

    @abstractmethod
    def load_model(self):
        pass

    def __init__(self, model_name, log_dir, options, train_eval_ds={},
                  pretrained_root_dir=None, merge_val_as_train=False):
        self.opt = options
        self.log_path = os.path.join(log_dir, model_name)

        self.models = {}  
        self.param_monodepth = []  
        self.param_pose_net = []
        
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
        
        self.optimizer_depth = optim.Adam(self.param_monodepth, self.opt.learning_rate, eps=1e-7, weight_decay=self.opt.weight_decay)
        self.lr_scheduler_depth = optim.lr_scheduler.StepLR(self.optimizer_depth, self.opt.scheduler_step_size, 0.1)


        train_dataset = train_eval_ds["train"]
        val_dataset = train_eval_ds["val"]
        
        
        if merge_val_as_train:
            print("\033[91m WARNING: Merging validation dataset into training dataset\033[0m")
            combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            self.train_loader = DataLoader(
            combined_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
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
    
    def set_train(self, train_position_only=False):
        """Convert models to training/eval mode based on position training flag"""
        # Define position-related models
        position_models = ["position_encoder", "position"]
        
        # Set requires_grad and model mode for each model
        for model_name in self.models:
            requires_grad = (not train_position_only)or(model_name in position_models)

            for param in self.models[model_name].parameters():
                param.requires_grad = requires_grad
            
            # Set model mode (train/eval)
            if requires_grad:
                self.models[model_name].train()
            else:
                self.models[model_name].eval()
                
    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        self.models["pose_encoder"].eval()
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

            self.optimizer_pose.zero_grad()
            loss_pose.backward()
            self.optimizer_pose.step()

            # Train depth
            self.set_train(train_position_only=False)
            outputs, losses = self.process_batch_depth(inputs)
            loss_depth = losses["loss"]

            self.optimizer_depth.zero_grad()
            loss_depth.backward()
            self.optimizer_depth.step()

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

                if self.opt.learn_intrinsics:
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

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            reprojection_loss = 0.9 * ms_ssim_loss + 0.1 * l1_loss

        return reprojection_loss

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
                        outputs[("refined", scale, frame_id)].detach()
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
                ).sum() / occu_mask_backward.sum()
                
            # Compute disparity smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            disp_smooth_loss = get_smooth_loss(norm_disp, color)

            # Combine losses
            loss += loss_reprojection / 2.0
            loss += self.opt.disparity_smoothness * disp_smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

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