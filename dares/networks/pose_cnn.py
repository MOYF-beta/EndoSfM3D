from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
    
class AttentionPoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(AttentionPoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        
        # 下采样网络 / Downsampling network
        self.downsample = nn.Sequential(
            nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3),
            nn.ReLU(True),
            AttentionBlock(16),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.ReLU(True),
            AttentionBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            AttentionBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            AttentionBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            AttentionBlock(256),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(True),
            AttentionBlock(256),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(True),
        )
        
        # 输出层 / Output layer
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.pose_conv(x)
        x = x.mean(3).mean(2)
        x = 0.01 * x.view(-1, self.num_input_frames - 1, 1, 6)
        
        axisangle = x[..., :3]
        translation = x[..., 3:]
        return axisangle, translation

class AttentionBlock(nn.Module):
    """通道注意力 + 空间注意力模块 / Channel attention + spatial attention module"""
    def __init__(self, in_channels, reduction=8):
        super(AttentionBlock, self).__init__()
        
        # 通道注意力 / Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 / Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return x * ca * sa
