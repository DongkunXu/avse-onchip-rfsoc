import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from .video_encoder import LightweightVideoEncoder
from .audio_encoder import TimedomainAudioEncoder
from .fusion import GatedCrossModalFusion
from utils.losses import ImprovedAVSELoss


class LightweightAVSE(pl.LightningModule):
    """轻量化AVSE模型 (HLS/FPGA友好版)
    
    该模型基于No.20的高性能版本进行修改，旨在平衡性能与FPGA实现友好性。
    它融合了No.20分支的先进融合策略和main分支的统一数据流架构。
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # 模型组件 (已全部更新为HLS友好版本)
        self.video_encoder = LightweightVideoEncoder(config)
        self.audio_encoder = TimedomainAudioEncoder(config)
        self.fusion = GatedCrossModalFusion(config)

        # 损失函数
        self.criterion = ImprovedAVSELoss(config.loss)

        # 学习率调度器参数
        self.lr = config.training.learning_rate
        self.warmup_steps = config.training.warmup_steps

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播 - 统一的 [B, C, T] 数据流架构"""
        video_frames = batch['video_frames']  # [B, T, H, W]
        mixed_audio = batch['mixed_audio']  # [B, 1, T_audio_in]

        # 1. 视觉特征提取
        video_features = self.video_encoder(video_frames)  # -> [B, T_video, C_video]

        # 2. 音频编码（返回 bottleneck 特征 + 各级 skip）
        audio_features, skips = self.audio_encoder.encode(mixed_audio)  # -> [B, C, T'], list

        # 3. 跨模态融合
        fused_features = self.fusion(audio_features, video_features)  # -> [B, C_audio, T']

        # 4. 音频解码（U-Net skip 由 encode 的 skips 提供）
        enhanced_audio = self.audio_encoder.decode(fused_features, skips)  # -> [B, 1, T]

        return enhanced_audio

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        enhanced_audio = self(batch)
        target_audio = batch['target_audio']

        # 统一长度：取最小长度
        min_length = min(enhanced_audio.shape[-1], target_audio.shape[-1])
        enhanced_audio = enhanced_audio[..., :min_length]
        target_audio = target_audio[..., :min_length]

        # 计算损失
        losses = self.criterion(enhanced_audio, target_audio)

        # 记录损失
        batch_size = enhanced_audio.shape[0]
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True, batch_size=batch_size)

        return losses['total_loss']

    def on_validation_epoch_start(self):
        """在验证周期开始时，重置日志记录相关的状态"""
        self.logged_scenes_this_epoch = set()
        self.scene_counts_this_epoch = {}
        self.logged_count_this_epoch = 0

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        enhanced_audio = self(batch)
        target_audio = batch['target_audio']

        # 统一长度：取最小长度
        min_length = min(enhanced_audio.shape[-1], target_audio.shape[-1])
        enhanced_audio = enhanced_audio[..., :min_length]
        target_audio = target_audio[..., :min_length]

        # 计算损失
        losses = self.criterion(enhanced_audio, target_audio)

        # 记录损失
        batch_size = enhanced_audio.shape[0]
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True, batch_size=batch_size)

        # 定期保存音频样本 (每个epoch都尝试)
        if self.current_epoch % 1 == 0:
            self._log_audio_samples(batch, enhanced_audio)

        return losses['total_loss']

    def on_validation_epoch_end(self):
        """在验证周期结束时，打印一次日志记录总结"""
        if self.current_epoch % 1 == 0 and self.logged_count_this_epoch > 0:
            print(f"\nEpoch {self.current_epoch} Summary: Logged {self.logged_count_this_epoch} audio samples from {len(self.logged_scenes_this_epoch)} scenes.")

    def _save_audio_sample(self, idx: int, scene_id: str, batch: Dict[str, torch.Tensor],
                           enhanced_audio: torch.Tensor):
        """保存单个音频样本的三种版本"""
        window_start = batch['window_start'][idx].item()

        # 记录增强音频
        self.logger.experiment.add_audio(
            f'epoch_{self.current_epoch}/enhanced_{scene_id}_start{window_start}',
            enhanced_audio[idx].detach().cpu(),
            global_step=self.global_step,
            sample_rate=16000
        )

        # 记录混合音频
        self.logger.experiment.add_audio(
            f'epoch_{self.current_epoch}/mixed_{scene_id}_start{window_start}',
            batch['mixed_audio'][idx].detach().cpu(),
            global_step=self.global_step,
            sample_rate=16000
        )

        # 记录目标音频
        self.logger.experiment.add_audio(
            f'epoch_{self.current_epoch}/target_{scene_id}_start{window_start}',
            batch['target_audio'][idx].detach().cpu(),
            global_step=self.global_step,
            sample_rate=16000
        )

    def _log_audio_samples(self, batch: Dict[str, torch.Tensor], enhanced_audio: torch.Tensor):
        """记录音频样本到TensorBoard - 使用epoch级别的计数器进行跨批次追踪"""
        if not hasattr(self.logger, 'experiment'):
            return

        max_scenes = 6  # 最多记录6个不同场景

        # 如果本周期已记录满，则提前退出
        if self.logged_count_this_epoch >= max_scenes:
            return

        scene_ids = batch['scene_id']

        for i in range(enhanced_audio.shape[0]):
            scene_id = scene_ids[i]

            # 更新本周期的场景出现次数
            self.scene_counts_this_epoch[scene_id] = self.scene_counts_this_epoch.get(scene_id, 0) + 1

            # 策略：当一个场景第二次出现时，并且它还没被记录过
            if self.scene_counts_this_epoch[scene_id] == 2 and scene_id not in self.logged_scenes_this_epoch:
                self._save_audio_sample(i, scene_id, batch, enhanced_audio)
                self.logged_scenes_this_epoch.add(scene_id)
                self.logged_count_this_epoch += 1

                # 如果记录满了，立即退出循环
                if self.logged_count_this_epoch >= max_scenes:
                    break

    def get_model_size(self):
        """计算模型大小"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.config.training.weight_decay,
            eps=1e-8
        )

        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.warmup_steps / total_steps,
            div_factor=10.0,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'val_total_loss'
            }
        }

class PlainAVSE(nn.Module):
    """
    A plain torch.nn.Module version of LightweightAVSE for inference and export.
    This class has the same architecture but does not inherit from pytorch_lightning.LightningModule,
    making it free from any Trainer-related dependencies.
    """
    def __init__(self, config):
        super().__init__()
        # No self.save_hyperparameters()
        self.config = config

        # Model components (copied from LightweightAVSE)
        self.video_encoder = LightweightVideoEncoder(config)
        self.audio_encoder = TimedomainAudioEncoder(config)
        self.fusion = GatedCrossModalFusion(config)

        # Note: Loss function and optimizer-related attributes are not needed for export.

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass (copied from LightweightAVSE)"""
        video_frames = batch['video_frames']
        mixed_audio = batch['mixed_audio']

        # 1. 视觉特征提取
        video_features = self.video_encoder(video_frames)

        # 2. 音频编码（返回 bottleneck 特征 + 各级 skip）
        audio_features, skips = self.audio_encoder.encode(mixed_audio)

        # 3. 跨模态融合
        fused_features = self.fusion(audio_features, video_features)

        # 4. 音频解码（U-Net skip 由 encode 的 skips 提供）
        enhanced_audio = self.audio_encoder.decode(fused_features, skips)

        return enhanced_audio
