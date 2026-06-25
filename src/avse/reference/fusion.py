import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossModalFusion(nn.Module):
    """Ablation: Concat-Projection Fusion (no gating).

    Replaces the 3-path gated fusion with a simple concat + 1x1 Conv baseline:
      audio_proj → channel_attention → align
      video_proj → channel_attention → upsample → align
      concat([audio_aligned, video_aligned], dim=C)
      1x1 Conv(2*fusion_dim → fusion_dim) + BN + ReLU
      residual + enhancement_net
      output_proj → audio_dim

    All other components (projections, channel attention, temporal smooth,
    enhancement_net, output_proj) are identical to the full model so that
    the only ablated variable is the gated multi-path fusion mechanism.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        video_dim = config.model.video_channels
        audio_dim = config.model.audio_channels
        fusion_dim = config.model.fusion_dim

        # Static temporal lengths at fusion input (derived from config).
        # Fixed pooling kernels replace AdaptiveAvgPool1d for HLS compatibility.
        audio_T = (config.audio.context_size + config.audio.future_samples) \
            // (2 ** len(config.model.kernel_sizes))
        video_T = config.video.frames_per_chunk + config.video.future_frames

        # Feature projection (identical to full model)
        self.audio_proj = nn.Sequential(
            nn.Conv1d(audio_dim, fusion_dim, 1, bias=False),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        self.video_proj = nn.Sequential(
            nn.Conv1d(video_dim, fusion_dim, 1, bias=False),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True)
        )

        # Video temporal alignment.
        # 2026-04-27 HW-friendly Tier B: dense Conv1d(C, C, k=5) (1317 DSP)
        # 替换为 DWSep (DW k=5 + PW 1×1), 大约 5× DSP/LUT 削减.
        # 2026-04-29 Tier D: upsample factor derived from audio_T / video_T
        # so the fusion length-alignment doesn't crop audio (when audio_layers
        # changes the bottleneck T, upsample factor must scale with it).
        # Tier C (audio_layers=6): audio_T=300, video_T=30 → factor=10.
        # Tier D (audio_layers=5): audio_T=600, video_T=30 → factor=20.
        self.video_upsample_factor = audio_T // video_T
        self.video_temporal_smooth = nn.Sequential(
            nn.Conv1d(fusion_dim, fusion_dim, kernel_size=5, padding=2,
                      groups=fusion_dim, bias=False),  # depthwise
            nn.Conv1d(fusion_dim, fusion_dim, 1, bias=True),  # pointwise (keeps the bias)
        )

        # Channel attention with fixed-kernel pooling (HLS-friendly static shape).
        self.audio_channel_attention = nn.Sequential(
            nn.AvgPool1d(kernel_size=audio_T, stride=audio_T),
            nn.Conv1d(fusion_dim, fusion_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_dim // 4, fusion_dim, 1),
            nn.Hardsigmoid(inplace=True)
        )
        self.video_channel_attention = nn.Sequential(
            nn.AvgPool1d(kernel_size=video_T, stride=video_T),
            nn.Conv1d(fusion_dim, fusion_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_dim // 4, fusion_dim, 1),
            nn.Hardsigmoid(inplace=True)
        )

        # Ablation: concat projection replaces all gate networks
        self.concat_proj = nn.Sequential(
            nn.Conv1d(fusion_dim * 2, fusion_dim, 1),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True)
        )

        # Tier D: enhancement_net + norm2 REMOVED.  At fusion_dim=64 the
        # 64→128→64 squeeze-expand was a heavy LUT consumer (~15% of fusion)
        # for a small accuracy gain.  Dropped along with the surrounding norm2
        # to give Vivado a fighting chance at LUT closure.
        # self.enhancement_net = ...   (REMOVED)
        # self.norm2 = ...             (REMOVED)

        # Pre-concat normalisation (kept — these are 1×C scale/bias only).
        self.norm_audio = nn.BatchNorm1d(fusion_dim)
        self.norm_video = nn.BatchNorm1d(fusion_dim)

        # Output projection (identical to full model)
        self.output_proj = nn.Conv1d(fusion_dim, audio_dim, 1)

    def forward(self, audio_features, video_features):
        """
        Args:
            audio_features: [B, C_audio, T_audio]
            video_features: [B, T_video, C_video]
        Returns:
            fused_features: [B, C_audio, T_audio]
        """
        # 0. Transpose video to [B, C_video, T_video]
        video_features_transposed = video_features.transpose(1, 2)

        # 1. Feature projection
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features_transposed)

        # 2. Channel attention
        audio_attn = self.audio_channel_attention(audio_proj)
        audio_attended = audio_proj * audio_attn

        video_attn = self.video_channel_attention(video_proj)
        video_attended = video_proj * video_attn

        # 3. Video upsample + temporal smooth
        video_up = video_attended.repeat_interleave(self.video_upsample_factor, dim=-1)
        video_aligned = self.video_temporal_smooth(video_up)

        # 4. Length alignment
        min_len = min(audio_attended.shape[-1], video_aligned.shape[-1])
        audio_aligned = audio_attended[..., :min_len]
        video_aligned = video_aligned[..., :min_len]

        # 5. Concat + 1x1 projection (ablation: replaces gated fusion)
        audio_norm = self.norm_audio(audio_aligned)
        video_norm = self.norm_video(video_aligned)

        concat_features = torch.cat([audio_norm, video_norm], dim=1)
        concat_fused = self.concat_proj(concat_features)

        # Residual connection with audio
        enhanced_audio = audio_aligned + concat_fused

        # 6. Output projection (Tier D: enhancement_net + norm2 dropped)
        output = self.output_proj(enhanced_audio)

        return output


CrossModalFusion = GatedCrossModalFusion
