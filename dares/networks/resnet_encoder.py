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
    

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module
    多头自注意力模块
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
            
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用1x1卷积进行线性变换 / Use 1x1 convolution for linear transformation
        # Use 1x1 convolution for linear transformation
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        # 添加层归一化 / Add layer normalization
        # Add layer normalization
        self.norm = nn.LayerNorm([channels])
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 首先进行LayerNorm，注意维度转换 / First perform LayerNorm, note the dimension conversion
        # First perform LayerNorm, note the dimension conversion
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 生成q, k, v / Generate q, k, v
        # Generate q, k, v
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3)  # 3, B, num_heads, H, W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数 / Calculate attention scores
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力分数
        # Apply attention scores
        x = (attn @ v).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MultiHeadAttentionalResnetEncoder(ResnetEncoder):
    """带多头注意力机制的ResNet编码器
    ResNet encoder with multi-head attention mechanism
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, num_heads=8):
        super(MultiHeadAttentionalResnetEncoder, self).__init__(num_layers, pretrained, num_input_images)
        
        # 初始化多头注意力模块列表
        # Initialize multi-head attention module list
        self.attentions = nn.ModuleList()
        for ch in self.num_ch_enc[1:]:  # 从layer1到layer4的输出通道
            # From layer1 to layer4 output channels
            self.attentions.append(MultiHeadAttention(ch, num_heads=num_heads))
        
        # 添加残差连接的辅助层
        # Add auxiliary layers for residual connection
        self.layer_norms = nn.ModuleList()
        for ch in self.num_ch_enc[1:]:    
            self.layer_norms.append(nn.LayerNorm([ch]))

    def forward(self, input_image):
        self.features = []
        x = input_image
        
        # 原始特征提取流程
        # Original feature extraction process
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        
        # 逐层处理并添加多头注意力
        # Process layer by layer and add multi-head attention
        x = self.encoder.maxpool(self.features[-1])
        for layer_idx in range(4):
            layer = getattr(self.encoder, f"layer{layer_idx+1}")
            x = layer(x)
            
            # 添加残差连接的多头注意力
            # Add multi-head attention with residual connection
            attention_out = self.attentions[layer_idx](x)
            x = x + attention_out  # 残差连接 / Residual connection
            
            self.features.append(x)
        
        return self.features