from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    支持不同输入图像数量的ResNet模型
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded = torch.hub.load_state_dict_from_url(models.ResNet18_Weights.IMAGENET1K_V1.url)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    ResNet编码器的Pytorch模块
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []
        # x = (input_image - 0.45) / 0.225
        # 原始特征提取流程 / Original feature extraction process
        # Original feature extraction process
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块
    Squeeze-and-Excitation attention module
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class AttentionalResnetEncoder(ResnetEncoder):
    """带注意力机制的ResNet编码器
    ResNet encoder with attention mechanism
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(AttentionalResnetEncoder, self).__init__(num_layers, pretrained, num_input_images)
        
        # 初始化注意力模块列表 / Initialize attention module list
        # Initialize attention module list
        self.attentions = nn.ModuleList()
        for ch in self.num_ch_enc[1:]:  # 从layer1到layer4的输出通道 / From layer1 to layer4 output channels
            # From layer1 to layer4 output channels
            self.attentions.append(SEBlock(ch))

    def forward(self, input_image):
        self.features = []
        x = input_image
        
        # 原始特征提取流程 / Original feature extraction process
        # Original feature extraction process
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        
        # 逐层处理并添加注意力 / Process layer by layer and add attention
        # Process layer by layer and add attention
        x = self.encoder.maxpool(self.features[-1])
        for layer_idx in range(4):
            layer = getattr(self.encoder, f"layer{layer_idx+1}")
            x = layer(x)
            x = self.attentions[layer_idx](x)  # 添加注意力 / Add attention
            self.features.append(x)
        
        return self.features
    

class PatchBasedMultiHeadAttention(nn.Module):
    """基于patch的多头注意力模块，生成attention map
    Patch-based multi-head attention module that generates attention maps
    """
    def __init__(self, channels, num_heads=8, patch_size=8, dropout=0.1):
        super(PatchBasedMultiHeadAttention, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        # embedding维度增加为resnet维度
        # Increase embedding dimension to resnet dimension
        self.embed_dim = channels
        
        # 输入投影层：将patch转换为embedding
        # Input projection layer: convert patch to embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, self.embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(num_groups=1, num_channels=self.embed_dim),
            nn.GELU()
        )
        
        # 使用PyTorch自带的多头注意力
        # Use PyTorch's native multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 使用(batch, seq_len, embed_dim)格式
        )
        
        # 位置编码（可学习）
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, self.embed_dim) * 0.02)  # 支持最大32x32的patch网格
        
        # 归一化层
        # Normalization layer
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 多次升采样和卷积来恢复原始尺寸
        # Multiple upsampling and convolution to restore original size
        self.upsample_layers = nn.ModuleList()
        current_dim = self.embed_dim
        
        # 计算需要的上采样次数（假设patch_size=8需要3次上采样：1->2->4->8）
        # Calculate required upsampling times (assuming patch_size=8 needs 3 upsamples: 1->2->4->8)
        num_upsample = int(np.log2(patch_size)) if patch_size > 1 else 0
        
        if num_upsample > 0:
            for i in range(num_upsample):
                # 每次上采样后减少通道数
                # Reduce channels after each upsampling
                next_dim = current_dim // 2 if i < num_upsample - 1 else channels
                
                self.upsample_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(current_dim, next_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=next_dim),
                    nn.GELU() if i < num_upsample - 1 else nn.Identity()
                ))
                current_dim = next_dim
        else:
            # 如果patch_size=1，直接降维到原始通道数
            # If patch_size=1, directly reduce to original channels
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(current_dim, channels, kernel_size=1),
                nn.GroupNorm(num_groups=1, num_channels=channels),
                nn.GELU()
            ))
        
        # 最终的attention map生成层
        # Final attention map generation layer
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 确保尺寸能被patch_size整除
        # Ensure dimensions are divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W
        
        # 分patch并投影到embedding空间
        # Split into patches and project to embedding space
        patch_embed = self.patch_embed(x)  # (B, embed_dim, H_patches, W_patches)
        
        # 重塑为序列格式
        # Reshape to sequence format
        B_new, embed_dim, H_patches, W_patches = patch_embed.shape
        num_patches = H_patches * W_patches
        
        # 展平patches: (B, embed_dim, H_patches, W_patches) -> (B, num_patches, embed_dim)
        # Flatten patches: (B, embed_dim, H_patches, W_patches) -> (B, num_patches, embed_dim)
        patch_tokens = patch_embed.view(B_new, embed_dim, num_patches).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 添加位置编码
        # Add positional encoding
        if num_patches <= self.pos_embed.size(1):
            pos_embed = self.pos_embed[:, :num_patches, :]
        else:
            # 如果patches数量超过预设，使用插值
            # If number of patches exceeds preset, use interpolation
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=num_patches, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        patch_tokens = patch_tokens + pos_embed
        
        # 归一化
        # Normalization
        patch_tokens = self.norm(patch_tokens)
        
        # 应用多头自注意力
        # Apply multi-head self-attention
        attn_output, _ = self.multihead_attn(patch_tokens, patch_tokens, patch_tokens)
        
        # 重塑回特征图格式: (B, num_patches, embed_dim) -> (B, embed_dim, H_patches, W_patches)
        # Reshape back to feature map format: (B, num_patches, embed_dim) -> (B, embed_dim, H_patches, W_patches)
        attn_output = attn_output.transpose(1, 2).view(B_new, embed_dim, H_patches, W_patches)
        
        # 通过多次升采样和卷积恢复到原始尺寸
        # Restore to original size through multiple upsampling and convolution
        for upsample_layer in self.upsample_layers:
            attn_output = upsample_layer(attn_output)
        
        # 裁剪回原始尺寸
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :, :H, :W]
        
        # 生成attention map
        # Generate attention map
        attention_map = self.attention_conv(attn_output)  # (B, 1, H, W)
        
        return attention_map


class MultiHeadAttentionalResnetEncoder(ResnetEncoder):
    """带多头注意力机制的ResNet编码器，生成attention map来强调重要区域
    ResNet encoder with multi-head attention mechanism that generates attention maps to emphasize important regions
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, num_heads=8, patch_size=8, 
                 attention_layers=None, attention_interval=2):
        super(MultiHeadAttentionalResnetEncoder, self).__init__(num_layers, pretrained, num_input_images)
        
        # 初始化多头注意力模块列表
        # Initialize multi-head attention module list
        self.attentions = nn.ModuleDict()
        
        # 确定使用注意力的层
        # Determine which layers to use attention
        if attention_layers is not None:
            # 如果明确指定了attention_layers，使用指定的层
            # If attention_layers is explicitly specified, use specified layers
            self.attention_layers = attention_layers
        else:
            # 否则使用间隔策略：每隔attention_interval层使用一次注意力
            # Otherwise use interval strategy: use attention every attention_interval layers
            self.attention_layers = []
            for i in range(4):  # ResNet有4个主要的layer (layer1-layer4)
                if i % attention_interval == 0:
                    self.attention_layers.append(i)
        
        # 为指定的层添加注意力模块
        # Add attention modules for specified layers
        for layer_idx in self.attention_layers:
            if layer_idx < len(self.num_ch_enc) - 1:  # 确保索引有效 / Ensure valid index
                self.attentions[f'layer_{layer_idx}'] = PatchBasedMultiHeadAttention(
                    channels=self.num_ch_enc[layer_idx + 1], 
                    num_heads=num_heads,
                    patch_size=patch_size
                )
        
        print(f"Attention layers: {self.attention_layers} (interval: {attention_interval})")

    def forward(self, input_image):
        self.features = []
        x = input_image
        
        # 原始特征提取流程
        # Original feature extraction process
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        
        # 逐层处理并在指定层添加注意力机制
        # Process layer by layer and add attention mechanism at specified layers
        x = self.encoder.maxpool(self.features[-1])
        for layer_idx in range(4):
            layer = getattr(self.encoder, f"layer{layer_idx+1}")
            x = layer(x)
            
            # 如果当前层需要使用注意力机制
            # If the current layer needs to use attention mechanism
            if layer_idx in self.attention_layers:
                attention_key = f'layer_{layer_idx}'
                if attention_key in self.attentions:
                    # 生成attention map
                    # Generate attention map
                    attention_map = self.attentions[attention_key](x)  # (B, 1, H, W)
                    
                    # 将attention map与特征图相乘，强调重要区域
                    # Multiply attention map with feature map to emphasize important regions
                    x = x * attention_map  # 广播相乘 / Broadcast multiplication
            
            self.features.append(x)
        
        return self.features