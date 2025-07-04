from __future__ import absolute_import, division, print_function

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class TransformDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4) , num_output_channels=3, use_skips=True):
        super(TransformDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict() # 有序字典
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("transform_conv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.Tanh = nn.Tanh()

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("transform", i)] = self.Tanh(self.convs[("transform_conv", i)](x))

        return self.outputs

from torch.nn import TransformerDecoderLayer
class VitTransformDecoder(nn.Module):
    def __init__(self, 
                 num_ch_enc,         # Encoder的通道数（ViT各层输出维度）
                 scales=range(4),    # 输出的多尺度层级（0为最高分辨率）
                 num_output_channels=3,
                 decoder_patch_size=16,
                 num_decoder_layers=1,
                 down_sample_method = 'interpolation'):
        super().__init__()
        
        self.scales = scales
        self.decoder_patch_size = decoder_patch_size
        self.num_output_channels = num_output_channels
        
        # Transformer解码器参数
        self.embed_dim = num_ch_enc[-1]  # 使用编码器最后一层的维度
        self.num_decoder_layers = num_decoder_layers
        
        # 关键组件 -------------------------------------------------
        # 1. Transformer解码层（轻量级）
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=self.embed_dim, 
                                   nhead=8,
                                   dim_feedforward=256,
                                   dropout=0.1)
            for _ in range(num_decoder_layers)
        ])
        
        # 2. 位置编码（与输入图像分辨率解耦）
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.embed_dim, 1, 1) * 0.02  # 可学习的全局位置编码
        )
        
        # 3. 图像块重组（生成最高分辨率图像）
        self.patch_reshape = nn.Sequential(
            nn.Conv2d(self.embed_dim, 
                     (decoder_patch_size**2) * num_output_channels,
                     kernel_size=1),  # 线性投影到像素空间
            nn.PixelShuffle(decoder_patch_size)  # 重组为图像
        )
        
        # 4. 多尺度下采样卷积头
        self.scale_convs = nn.ModuleDict()
        for s in self.scales:
            if down_sample_method == 'interpolation':
            # 使用插值下采样
                head = nn.Sequential(
                    nn.Upsample(scale_factor=1/(2), mode='bilinear', align_corners=True),
                )
            elif down_sample_method == 'conv':
            # 使用卷积下采样
                head = nn.Sequential(
                    nn.Conv2d(num_output_channels, num_output_channels, 
                        kernel_size=3, stride=2, padding=1),  # 下采样
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_output_channels, num_output_channels, 
                        kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                # 初始化最后一层卷积的权重
                head[-2].weight.data.normal_(0, 1e-5)
                head[-2].bias.data.zero_()
            self.scale_convs[str(s)] = head
        # 5. 激活函数
        self.Tanh = nn.Tanh()
    def forward(self, input_features):
        """
        输入特征：list[Tensor]，来自编码器的各层输出
        假设最后一层特征形状为 [B, C, H, W]（ViT调整后的空间格式）
        """
        x = input_features[-1]  # 只使用最后一层特征
        
        # 添加可学习的全局位置编码
        x = x + self.pos_embed
        
        # Transformer解码处理 -------------------------------------
        # 转换为序列格式 [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        
        # 逐层处理（自注意力）
        for layer in self.decoder_layers:
            x = layer(x, x)
        
        # 转换回空间格式 [B, N, C] -> [B, C, H, W]
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        # 图像块重组 ----------------------------------------------
        x = self.patch_reshape(x)  # [B, num_output_channels, H*patch, W*patch]
        
        # 生成多尺度输出 ------------------------------------------
        outputs = {}
        for s in self.scales:
            if s == 0:  # 最高分辨率（原始分辨率）
                outputs[("transform", s)] = self.Tanh(x)
            else:       # 下采样到更低分辨率
                # 逐级下采样
                scale_output = x
                for _ in range(s):
                    scale_output = self.scale_convs[str(s)](scale_output)
                outputs[("transform", s)] = self.Tanh(scale_output)
        
        return outputs
