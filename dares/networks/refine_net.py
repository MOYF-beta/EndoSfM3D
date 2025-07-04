import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ca(x)  # 通道注意力
        x = self.sa(x)  # 空间注意力
        x += residual
        x = self.relu(x)
        return x

class Refine_net(nn.Module):
    def __init__(self, num_blocks=8, feat_channels=64):
        super().__init__()
        # 输入通道：3 (img) + 1 (disp) = 4
        self.input_conv = nn.Sequential(
            nn.Conv2d(4, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        
        # 堆叠残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(feat_channels) for _ in range(num_blocks)]
        )
        
        # 多尺度特征融合
        self.pyramid_conv = nn.ModuleList([
            nn.Conv2d(feat_channels, feat_channels//2, 3, padding=1, dilation=1),
            nn.Conv2d(feat_channels, feat_channels//2, 3, padding=2, dilation=2),
            nn.Conv2d(feat_channels, feat_channels//2, 3, padding=4, dilation=4)
        ])
        
        # 特征融合层
        self.pyramid_fusion = nn.Conv2d(feat_channels + (feat_channels//2)*3, feat_channels, 1)
        
        # 最终预测
        self.output_conv = nn.Conv2d(feat_channels, 1, 3, padding=1)

    def forward(self, disp, img):
        # 尺寸对齐
        if img.shape[-2:] != disp.shape[-2:]:
            img = F.interpolate(img, size=disp.shape[-2:], mode='bilinear', align_corners=False)
        
        # 拼接输入
        x = torch.cat((img, disp), dim=1)
        
        # 初始特征提取
        x = self.input_conv(x)
        
        # 残差块处理
        res_feat = self.res_blocks(x)
        
        # 多尺度上下文融合
        p1 = self.pyramid_conv[0](res_feat)
        p2 = self.pyramid_conv[1](res_feat)
        p3 = self.pyramid_conv[2](res_feat)
        pyramid_feat = torch.cat([p1, p2, p3], dim=1)
        
        # 融合特征
        fused_feat = self.pyramid_fusion(torch.cat([res_feat, pyramid_feat], dim=1))
        
        # 残差预测
        residual = self.output_conv(fused_feat)
        return disp + residual

# 示例用法
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disp = torch.randn(2, 1, 320, 256).to(device)
    img = torch.randn(2, 3, 320, 256).to(device)
    model = Refine_net().to(device)
    output = model(disp, img)
    print(f"Input disp shape: {disp.shape}")
    print(f"Output shape: {output.shape}")