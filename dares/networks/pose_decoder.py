from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        # Ensure model layers are on the same device as input
        device = last_features[0].device
        self.convs["squeeze"] = self.convs["squeeze"].to(device)
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

class PoseDecoder_with_intrinsics(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None):
        super(PoseDecoder_with_intrinsics, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if predict_intrinsics:
            assert image_width is not None and image_height is not None\
                , "image_width and image_height must be provided if predict_intrinsics is True"
            self.image_width = image_width
            self.image_height = image_height

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # if predict_intrinsics, the feature is extracted here
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.softlpus = nn.Softplus()
        if self.predict_intrinsics:
            if simplified_intrinsic:
                # fx, fy = ? ; cx = cy = 0.5
                self.num_param_to_predict = 2
            else:
                # fx, fy, cx, cy = ?
                self.num_param_to_predict = 4
            
            self.intrinsics_layers = nn.Sequential(
                ConvBlock(256, 256),
                ConvBlock(256, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.num_param_to_predict),
                
            )


        self.net = nn.ModuleList(list(self.convs.values()))
    def forward(self, input_features):
        def predict_intrinsics(feature_for_intrinsics):
            batch_size = feature_for_intrinsics.shape[0]
            device = feature_for_intrinsics.device
            # prepare intrinsics matrix
            intrinsics_mat = torch.eye(4, device=device).unsqueeze(0)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            # do the prediction
            intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            # construct the intrinsics matrix
            foci = (intrinsics[:, :2] + 0.5) * torch.tensor([self.image_width, self.image_height], device=device)
            foci_mat = self.softlpus(torch.diag_embed(foci))
            if self.num_param_to_predict == 4:
                offsets = (intrinsics[:, 2:] + 0.5) * torch.tensor([self.image_width, self.image_height], device=device)
            else:
                offsets = torch.ones((batch_size,2), device=device) * 0.5
            intrinsics_mat[:, :2, :2] = foci_mat
            intrinsics_mat[:, :2, 2:3] = offsets.unsqueeze(-1)

            return intrinsics_mat

        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
            # if i == 0 and self.predict_intrinsics:
            #     feature_for_intrinsics = out

        out = out.mean(3).mean(2)
        # NOTE : Here is a key issue of training on different datasets. On previous studies this is fixed
        # Beili says this vary from dataset to dataset.
        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]
        if self.predict_intrinsics:
            return axisangle, translation, predict_intrinsics(cat_features)
        else:
            return axisangle, translation

# class PoseDecoder_with_intrinsics(nn.Module):
#     def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
#                  predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None, auto_scale=True):
#         super(PoseDecoder_with_intrinsics, self).__init__()
#         self.auto_scale = auto_scale
#         self.num_ch_enc = num_ch_enc
#         self.num_input_features = num_input_features
#         if predict_intrinsics:
#             assert image_width is not None and image_height is not None\
#                 , "image_width and image_height must be provided if predict_intrinsics is True"
#             self.image_width = image_width
#             self.image_height = image_height

#         if num_frames_to_predict_for is None:
#             num_frames_to_predict_for = num_input_features - 1
#         self.num_frames_to_predict_for = num_frames_to_predict_for
#         self.predict_intrinsics = predict_intrinsics

#         self.convs = OrderedDict()
#         self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
#         self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
#         self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)        # if predict_intrinsics, the feature is extracted here
#         self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

#         self.relu = nn.ReLU()
#         self.softlpus = nn.Softplus()
        
#         if self.predict_intrinsics:
#             if simplified_intrinsic:
#                 # fx, fy = ? ; cx = cy = 0.5
#                 self.num_param_to_predict = 2
#             else:
#                 # fx, fy, cx, cy = ?
#                 self.num_param_to_predict = 4
            
#             self.intrinsics_layers = nn.Sequential(
#                 ConvBlock(256, 256),
#                 ConvBlock(256, 64),
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(64, self.num_param_to_predict),
#             )
        
#         # 添加scale_factor预测分支
#         if self.auto_scale:
#             self.scale_factor_layers = nn.Sequential(
#                 ConvBlock(256, 128),
#                 ConvBlock(128, 64),
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(64, 1),
#                 nn.Sigmoid()  # 输出0-1范围，后续会映射到1e-2到1e2
#             )
        
#         self.net = nn.ModuleList(list(self.convs.values()))
#     def forward(self, input_features):
#         def predict_intrinsics(feature_for_intrinsics):
#             batch_size = feature_for_intrinsics.shape[0]
            
#             # prepare intrinsics matrix
#             intrinsics_mat = torch.eye(4, device=device).unsqueeze(0)
#             intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
#             # do the prediction
#             intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            
#             # Apply sigmoid to normalize predictions to [0, 1] range for numerical stability
#             intrinsics_normalized = torch.sigmoid(intrinsics)
            
#             # Scale focal lengths to reasonable range [0.5, 2.5] times image dimensions
#             # This ensures positive values and prevents singularity
#             foci_scale_min, foci_scale_max = 0.5, 2.5
#             foci_normalized = intrinsics_normalized[:, :2]
#             foci_scaled = foci_scale_min + foci_normalized * (foci_scale_max - foci_scale_min)
#             foci = foci_scaled * torch.tensor([self.image_width, self.image_height], device=device)
            
#             # Apply softplus and add minimum threshold to ensure positive focal lengths
#             foci = self.softlpus(foci) + 1e-3  # Minimum focal length to prevent singularity
#             if self.num_param_to_predict == 4:
#                 # For full intrinsics, ensure principal points are within valid range
#                 offsets_normalized = intrinsics_normalized[:, 2:]
#                 # Map to [0.1, 0.9] of image dimensions to avoid edge cases
#                 offsets = (0.1 + offsets_normalized * 0.8) * torch.tensor([self.image_width, self.image_height], device=device)
#             else:
#                 # For simplified intrinsics, set principal points to image center with proper scaling
#                 offsets = torch.ones((batch_size, 2), device=device) * torch.tensor([self.image_width / 2.0, self.image_height / 2.0], device=device)
            
#             # Construct intrinsics matrix safely with bounds checking
#             intrinsics_mat[:, 0, 0] = torch.clamp(foci[:, 0], min=1e-3)  # fx - ensure positive
#             intrinsics_mat[:, 1, 1] = torch.clamp(foci[:, 1], min=1e-3)  # fy - ensure positive  
#             intrinsics_mat[:, 0, 2] = torch.clamp(offsets[:, 0], min=1.0, max=self.image_width-1.0)  # cx
#             intrinsics_mat[:, 1, 2] = torch.clamp(offsets[:, 1], min=1.0, max=self.image_height-1.0)  # cy
#             return intrinsics_mat

#         last_features = [f[-1] for f in input_features]

#         cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
#         cat_features = torch.cat(cat_features, 1)
#         out = cat_features
#         for i in range(3):
#             out = self.convs[("pose", i)](out)
#             if i != 2:
#                 out = self.relu(out)
#             # if i == 0 and self.predict_intrinsics:
#             #     feature_for_intrinsics = out

#         out = out.mean(3).mean(2)
        
#         # 预测scale_factor (从1e-2到1e2的范围)
#         scale_factor = self.scale_factor_layers(cat_features) if self.auto_scale else torch.ones(out.shape[0], 1, device=device)
#         # 将0-1范围映射到1e-2到1e2 (log scale)
#         scale_factor = torch.exp(scale_factor * (torch.log(torch.tensor(1e3)) - torch.log(torch.tensor(1e-3))) + torch.log(torch.tensor(1e-3)))
        
#         # NOTE : Here is a key issue of training on different datasets. On previous studies this is fixed
#         # Beili says this vary from dataset to dataset.
#         # 应用通过网络估计的scale_factor
#         out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6) * scale_factor.unsqueeze(1).unsqueeze(2)

#         axisangle = out[..., :3]
#         translation = out[..., 3:]
#         if self.predict_intrinsics:
#             return axisangle, translation, predict_intrinsics(cat_features)
#         else:
#             return axisangle, translation


class SimplifiedPoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None,
                 predict_intrinsics=False, image_width=None, image_height=None, auto_scale=False):
        super(SimplifiedPoseDecoder, self).__init__()
        
        # Basic initialization
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.predict_intrinsics = predict_intrinsics
        
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        
        if predict_intrinsics:
            self.image_width = image_width
            self.image_height = image_height
        
        # Simplified convolution layers
        self.convs = OrderedDict()
        # Reduce initial channels to 128 instead of 256
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 128, 1)
        # Simplified pose prediction pathway
        self.convs["pose_pred"] = nn.Sequential(
            nn.Conv2d(num_input_features * 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 6 * num_frames_to_predict_for, 1)
        )
        
        # Simplified intrinsics prediction (if enabled)
        if self.predict_intrinsics:
            self.intrinsics_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_input_features * 128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 4)  # Predicting fx, fy, cx, cy
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, input_features):
        # Process input features
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        # Pose prediction
        pose_out = self.convs["pose_pred"](cat_features)
        pose_out = pose_out.mean(3).mean(2)
        pose_out = 0.001 * pose_out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        # Split pose into rotation and translation
        axisangle = pose_out[..., :3]
        translation = pose_out[..., 3:]
        
        # Intrinsics prediction (if enabled)
        if self.predict_intrinsics:
            batch_size = cat_features.shape[0]
            intrinsics = self.intrinsics_layers(cat_features)
            
            # Construct intrinsics matrix
            intrinsics_mat = torch.eye(4).unsqueeze(0).to(device)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            # Scale predictions to image dimensions
            foci = (intrinsics[:, :2] + 0.5) * torch.Tensor(
                [self.image_width, self.image_height],
                
            ).to(device)
            offsets = (intrinsics[:, 2:] + 0.5) * torch.Tensor(
                [self.image_width, self.image_height],
            ).to(device)
            
            # Fill intrinsics matrix
            intrinsics_mat[:, 0, 0] = foci[:, 0]  # fx
            intrinsics_mat[:, 1, 1] = foci[:, 1]  # fy
            intrinsics_mat[:, 0, 2] = offsets[:, 0]  # cx
            intrinsics_mat[:, 1, 2] = offsets[:, 1]  # cy
            
            return axisangle, translation, intrinsics_mat
        
        return axisangle, translation


class KNNFeatureProcessor(nn.Module):
    """KNN-based feature processing module for pose estimation"""
    def __init__(self, feature_dim, pose_dim=6, k=5, temperature=1.0):
        super(KNNFeatureProcessor, self).__init__()
        self.k = k
        self.temperature = temperature
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # Feature bank for storing reference features (can be updated during training)
        self.register_buffer('feature_bank', torch.randn(1000, feature_dim))
        self.register_buffer('pose_bank', torch.randn(1000, pose_dim))  # 对应的pose参数
        self.register_buffer('bank_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('bank_size', torch.zeros(1, dtype=torch.long))
        
        # Learnable parameters for feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def update_feature_bank(self, features, poses):
        """Update the feature bank with new features and corresponding poses"""
        batch_size = features.size(0)
        
        # 更新feature bank
        ptr = int(self.bank_ptr)
        bank_capacity = self.feature_bank.size(0)
        
        if ptr + batch_size <= bank_capacity:
            self.feature_bank[ptr:ptr + batch_size] = features.detach()
            self.pose_bank[ptr:ptr + batch_size] = poses.detach()
            self.bank_ptr[0] = (ptr + batch_size) % bank_capacity
        else:
            # Circular buffer behavior
            remaining = bank_capacity - ptr
            self.feature_bank[ptr:] = features[:remaining].detach()
            self.pose_bank[ptr:] = poses[:remaining].detach()
            self.feature_bank[:batch_size - remaining] = features[remaining:].detach()
            self.pose_bank[:batch_size - remaining] = poses[remaining:].detach()
            self.bank_ptr[0] = batch_size - remaining
            
        self.bank_size[0] = min(self.bank_size[0] + batch_size, bank_capacity)
        
    def knn_search(self, query_features):
        """Perform KNN search in the feature bank"""
        if self.bank_size[0] == 0:
            return None, None
            
        # 计算相似度
        query_features_norm = F.normalize(query_features, dim=1)
        bank_features_norm = F.normalize(self.feature_bank[:self.bank_size[0]], dim=1)
        
        # 计算余弦相似度
        similarities = torch.mm(query_features_norm, bank_features_norm.t())
        
        # 找到top-k最相似的特征
        k = min(self.k, self.bank_size[0])
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1)
        
        # 应用温度缩放并计算权重
        weights = F.softmax(top_k_similarities / self.temperature, dim=1)
        
        # 获取对应的特征和pose
        batch_size = query_features.size(0)
        neighbor_features = self.feature_bank[top_k_indices.view(-1)].view(batch_size, k, -1)
        neighbor_poses = self.pose_bank[top_k_indices.view(-1)].view(batch_size, k, -1)
        
        # 加权聚合邻居特征
        weighted_neighbor_features = torch.sum(
            neighbor_features * weights.unsqueeze(-1), dim=1
        )
        
        # 加权聚合邻居pose（用于参考）
        weighted_neighbor_poses = torch.sum(
            neighbor_poses * weights.unsqueeze(-1), dim=1
        )
        
        return weighted_neighbor_features, weighted_neighbor_poses
        
    def forward(self, features):
        """Process features using KNN"""
        # Flatten features for KNN processing
        batch_size = features.size(0)
        original_shape = features.shape
        flattened_features = features.view(batch_size, -1)
        
        # Perform KNN search
        neighbor_features, _ = self.knn_search(flattened_features)
        
        if neighbor_features is not None:
            # Fuse original features with neighbor features
            fused_input = torch.cat([flattened_features, neighbor_features], dim=1)
            processed_features = self.fusion_layer(fused_input)
            
            # Reshape back to original feature map shape
            processed_features = processed_features.view(original_shape)
        else:
            processed_features = features
            
        return processed_features


class KNNPoseDecoder_with_intrinsics(nn.Module):
    """Enhanced PoseDecoder with KNN feature processing for improved pose estimation"""
    
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None, 
                 auto_scale=True, use_knn=True, knn_k=5, knn_temperature=1.0):
        super(KNNPoseDecoder_with_intrinsics, self).__init__()
        self.auto_scale = auto_scale
        self.use_knn = use_knn
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        
        if predict_intrinsics:
            assert image_width is not None and image_height is not None, \
                "image_width and image_height must be provided if predict_intrinsics is True"
            self.image_width = image_width
            self.image_height = image_height

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
          # 添加KNN特征处理器
        if self.use_knn:
            # 计算特征维度 (考虑feature map的spatial dimensions)
            # 假设feature map经过global pooling后的维度
            feature_dim = num_input_features * 256  # 基础特征维度
            pose_dim = 6 * num_frames_to_predict_for  # pose维度
            self.knn_processor = KNNFeatureProcessor(
                feature_dim=feature_dim, 
                pose_dim=pose_dim,
                k=knn_k, 
                temperature=knn_temperature
            )
        
        if self.predict_intrinsics:
            if simplified_intrinsic:
                # fx, fy = ? ; cx = cy = 0.5
                self.num_param_to_predict = 2
            else:
                # fx, fy, cx, cy = ?
                self.num_param_to_predict = 4
            
            self.intrinsics_layers = nn.Sequential(
                ConvBlock(256, 256),
                ConvBlock(256, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.num_param_to_predict),
            )
        
        # 添加scale_factor预测分支
        if self.auto_scale:
            self.scale_factor_layers = nn.Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出0-1范围，后续会映射到1e-2到1e2
            )
        
        self.net = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, input_features):
        def predict_intrinsics(feature_for_intrinsics):
            batch_size = feature_for_intrinsics.shape[0]
            device = feature_for_intrinsics.device
            
            # prepare intrinsics matrix
            intrinsics_mat = torch.eye(4, device=device).unsqueeze(0)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            # do the prediction
            intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            
            # Apply sigmoid to normalize predictions to [0, 1] range for numerical stability
            intrinsics_normalized = torch.sigmoid(intrinsics)
            
            # Scale focal lengths to reasonable range [0.5, 2.5] times image dimensions
            # This ensures positive values and prevents singularity
            foci_scale_min, foci_scale_max = 0.5, 2.5
            foci_normalized = intrinsics_normalized[:, :2]
            foci_scaled = foci_scale_min + foci_normalized * (foci_scale_max - foci_scale_min)
            foci = foci_scaled * torch.tensor([self.image_width, self.image_height], device=device)
            
            # Apply softplus and add minimum threshold to ensure positive focal lengths
            foci = self.softplus(foci) + 1e-3  # Minimum focal length to prevent singularity
            if self.num_param_to_predict == 4:
                # For full intrinsics, ensure principal points are within valid range
                offsets_normalized = intrinsics_normalized[:, 2:]
                # Map to [0.1, 0.9] of image dimensions to avoid edge cases
                offsets = (0.1 + offsets_normalized * 0.8) * torch.tensor([self.image_width, self.image_height], device=device)
            else:
                # For simplified intrinsics, set principal points to image center with proper scaling
                offsets = torch.ones((batch_size, 2), device=device) * torch.tensor([self.image_width / 2.0, self.image_height / 2.0], device=device)
            
            # Construct intrinsics matrix safely with bounds checking
            intrinsics_mat[:, 0, 0] = torch.clamp(foci[:, 0], min=1e-3)  # fx - ensure positive
            intrinsics_mat[:, 1, 1] = torch.clamp(foci[:, 1], min=1e-3)  # fy - ensure positive  
            intrinsics_mat[:, 0, 2] = torch.clamp(offsets[:, 0], min=1.0, max=self.image_width-1.0)  # cx
            intrinsics_mat[:, 1, 2] = torch.clamp(offsets[:, 1], min=1.0, max=self.image_height-1.0)  # cy
            return intrinsics_mat

        last_features = [f[-1] for f in input_features]
        device = last_features[0].device

        # 确保网络层在正确设备上
        for key, layer in self.convs.items():
            self.convs[key] = layer.to(device)

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        # 应用KNN处理 (在特征提取后，pose预测前)
        if self.use_knn:
            # 对concatenated features进行KNN处理
            # 先进行一次pooling得到global特征用于KNN
            pooled_features = F.adaptive_avg_pool2d(cat_features, 1).view(cat_features.size(0), -1)
            enhanced_pooled_features = self.knn_processor(pooled_features.unsqueeze(-1).unsqueeze(-1))
            enhanced_pooled_features = enhanced_pooled_features.view(pooled_features.size(0), -1)
            
            # 将增强的全局特征信息融合回原始特征图
            # 这里使用简单的广播加法，您也可以设计更复杂的融合策略
            enhanced_features = enhanced_pooled_features.unsqueeze(-1).unsqueeze(-1)
            enhanced_features = enhanced_features.expand_as(cat_features)
            
            # 特征融合：原始特征 + KNN增强特征
            cat_features = cat_features + 0.1 * enhanced_features  # 使用较小的权重避免主导
        
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        
        # 预测scale_factor (从1e-2到1e2的范围)
        if self.auto_scale:
            scale_factor = self.scale_factor_layers(cat_features)
            # 将0-1范围映射到1e-2到1e2 (log scale)
            scale_factor = torch.exp(scale_factor * (torch.log(torch.tensor(1e2, device=device)) - torch.log(torch.tensor(1e-2, device=device))) + torch.log(torch.tensor(1e-2, device=device)))
        else:
            scale_factor = torch.ones(out.shape[0], 1, device=device)
        
        # NOTE : Here is a key issue of training on different datasets. On previous studies this is fixed
        # Beili says this vary from dataset to dataset.
        # 应用通过网络估计的scale_factor
        out = 0.001 * out.view(-1, self.num_frames_to_predict_for, 1, 6) * scale_factor.unsqueeze(1).unsqueeze(2)

        axisangle = out[..., :3]
        translation = out[..., 3:]        # 更新KNN特征库 (在训练时)
        if self.use_knn and self.training:
            # 提取当前batch的特征和预测的pose用于更新特征库
            current_features = F.adaptive_avg_pool2d(cat_features, 1).view(cat_features.size(0), -1)
            current_poses = out.view(out.size(0), -1)  # flatten pose predictions
            self.knn_processor.update_feature_bank(current_features, current_poses)
        
        if self.predict_intrinsics:
            return axisangle, translation, predict_intrinsics(cat_features)
        else:
            return axisangle, translation