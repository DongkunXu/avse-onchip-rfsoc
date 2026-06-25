from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class DataConfig:
    """数据配置"""
    root_dir: str = "path/to/dataset"
    batch_size: int = 24
    num_workers: int = 6
    pin_memory: bool = True
    init_workers: int = 14
    prefetch_factor: int = 2

    # Window settings for dataset preparation
    window_duration: float = 1.20  # 1200ms total window for enhanced context
    overlap_ratio: float = 0.20    # 20% overlap between windows


@dataclass
class VideoConfig:
    """视频处理配置"""
    fps: int = 25
    input_size: List[int] = field(default_factory=lambda: [96, 96])
    source_size: List[int] = field(default_factory=lambda: [224, 224])
    frames_per_chunk: int = 29  # 1160ms (1.2s - 40ms future)
    future_frames: int = 1  # 未来1帧 (40ms) - 在50ms延迟预算内


@dataclass
class AudioConfig:
    """音频处理配置"""
    sample_rate: int = 16000
    chunk_size: int = 800  # 50ms音频块
    context_size: int = 18560  # 1160ms上下文窗口
    future_samples: int = 640  # 40ms延迟预算中的未来样本


@dataclass
class ModelConfig:
    """模型配置 — 2026-04-29 HW-friendly Tier D 默认值.

    Tier D 是为 ZU48DR LUT/BRAM 装下做的最后一轮压缩。Vivado P&R 实测
    Tier C (DSP 57% / BRAM 152% / LUT 430%) 仍然 LUT 不可能 fit。
    Tier D 在 Tier C 之上：

      - audio_layers 6 → 5 (砍最深一层, audio_channels 256 → 192)
        节省: audio_dec BRAM 从 11 个 ping-pong 减到 9 个; bottleneck 维度
        小一档. 注意 T 在 bottleneck 翻倍 (300 → 600), 但 channel 维少 25%
        + 少一对 main/out buffer = 净 BRAM -25%.
      - kernel_sizes 25% 缩 (整数, 平滑递减): [25,19,15,11,7,5] →
        [19,15,11,9,7] (现在 5 项匹配 audio_layers=5).
      - video_layers 4 → 3 (砍 V4 stage), video_channels 192 → 128
        节省: video LUT/DSP/BRAM 各 ~-40%.
      - fusion_dim 128 → 64
        节省: fusion 内部所有 ops O(C²) → -75%.
      - 去掉 fusion enhancement_net + norm2 (在 fusion forward 中)
        节省: 2 个 1×1 conv (C→2C→C) + 1 个 BN. fusion -15%.

    SI-SDR 累积估损 ~3 dB (Tier C 的 1.5 + Tier D 增量 ~1.5).
    skip connection 仍然完整保留低频细节, U-Net 主路径不变.
    """
    # 视觉编码器: 192 → 128 → 96 (Tier E.1: -25% channels for DFX RR fit)
    # Tier E.1 motivation: V6 linear_relu (CIN=128*4=512 → 128) was 30% chip
    # LUT alone; channels² scaling drops it to ~17% at 96.  Whole video IP
    # estimated 95% → 59% LUT, fits a DFX RR with margin.
    video_channels: int = 96
    video_layers: int = 3

    # 音频编码器: 256 → 192, 6 → 5 layers (Tier D)
    # kernel_sizes 25% 缩, 5 项匹配 audio_layers=5.
    audio_channels: int = 192
    audio_layers: int = 5
    kernel_sizes: List[int] = field(default_factory=lambda: [19, 15, 11, 9, 7])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])

    # 融合模块: 128 → 64 (Tier D, 4× compute cut vs Tier C)
    fusion_dim: int = 64
    fusion_heads: int = 4
    fusion_dropout: float = 0.08


@dataclass
class TrainingConfig:
    """训练配置"""
    max_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 2e-4
    gradient_clip: float = 1.0
    warmup_steps: int = 4000


@dataclass
class LossConfig:
    """损失函数配置（更新为改进版本）"""
    # 基础损失权重
    l1_weight: float = 1.0
    l2_weight: float = 1.0

    # 感知损失
    stoi_weight: float = 5.0
    pesq_weight: float = 0.8

    # 多尺度时域损失
    multiscale_weight: float = 1.0

    si_sdr_weight: float = 0.5


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_yaml(cls, path: str):
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config