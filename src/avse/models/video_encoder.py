import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """深度可分离卷积（适合DPU加速）+ 残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接：当输入输出通道不同或stride>1时需要projection
        self.use_residual = True
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 保存输入用于残差连接
        residual = self.shortcut(x)
        
        # 主分支
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        
        # 残差连接 (DPU支持的element-wise add)
        out = out + residual
        return out


class LightweightVideoEncoder(nn.Module):
    """轻量化视频编码器（针对HLS/FPGA优化）"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config.model.video_channels

        # 空间特征提取 - 根据配置动态构建
        # Spatial trace (96x96 input, V0 stride 2 then num_layers DWSep stride 2):
        #   num_layers=4: 96 -> 48 -> 24 -> 12 -> 6 -> 3 (4 DWSep) -> 2 (AvgPool k=2) -> 1 (Conv k=2)
        #   num_layers=3 (Tier D): 96 -> 48 -> 24 -> 12 -> 6 (3 DWSep) -> 2 (AvgPool k=5) -> 1 (Conv k=2)
        layers = []
        in_channels = 1

        # 第一层：固定Conv2d (96x96 -> 48x48)
        layers.extend([
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ])
        in_channels = 64

        # 动态构建DepthwiseConv层
        num_layers = config.model.video_layers
        for i in range(num_layers):
            # 渐进增加通道数：64 -> 128 -> 192 -> 256
            out_channels = min(64 + (i + 1) * 64, channels)

            layers.append(
                DepthwiseSeparableConv2d(in_channels, out_channels, 3, stride=2, padding=1)
            )
            in_channels = out_channels

        # === HLS 优化：固定核 AvgPool2d (替代 AdaptiveAvgPool2d) ===
        # 动态计算 AvgPool 内核以保证 feature_proj (k=2) 后输出 1x1.
        # 96x96 输入: V0 stride 2 -> 48; 然后每个 DWSep stride 2 减半.
        # spatial after V0+num_layers DWSep = 96 / 2^(num_layers+1)
        #   num_layers=4: spatial=3 -> AvgPool k=2 -> 2x2 -> Conv k=2 -> 1x1
        #   num_layers=3: spatial=6 -> AvgPool k=5 -> 2x2 -> Conv k=2 -> 1x1
        spatial_after_dwsep = 96 // (2 ** (num_layers + 1))
        avgpool_k = spatial_after_dwsep - 1   # 让 AvgPool 输出 = 2x2
        layers.append(nn.AvgPool2d(kernel_size=avgpool_k, stride=1))

        self.spatial_encoder = nn.Sequential(*layers)

        # Feature compression: Conv2d(k=2) on 2x2 feature map flattens spatial
        # to 1x1 (equivalent to Linear(C*4 -> C)). HLS-friendly: keeps the
        # entire video path as Conv2d, no dense matmul kernel required.
        self.feature_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 时序处理（简化的注意力）
        self.temporal_proj = nn.Linear(channels, channels)

    def forward(self, video_frames):
        """
        Args:
            video_frames: [B, T, H, W] = [B, T, 96, 96]
        Returns:
            features: [B, T, C] = [B, 7, 256]
        """
        B, T, H, W = video_frames.shape

        # 重塑为 [B*T, 1, H, W] 以便批量处理
        x = video_frames.unsqueeze(2)  # [B, T, 1, H, W]
        x = x.view(B * T, 1, H, W)  # [B*T, 1, H, W]

        # 空间特征提取
        spatial_features = self.spatial_encoder(x)  # [B*T, C, 2, 2]

        # 特征投影 (Conv2d k=2 -> 1x1 spatial)
        features = self.feature_proj(spatial_features)  # [B*T, C, 1, 1]
        features = features.view(B * T, -1)  # [B*T, C]

        # 重塑回时序形式
        features = features.view(B, T, -1)  # [B, T, C]

        # 简单的时序处理（移除LayerNorm以提高DPU兼容性）
        temporal_features = self.temporal_proj(features)
        temporal_features = temporal_features + features  # 直接残差连接

        return temporal_features
