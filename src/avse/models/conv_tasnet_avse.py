"""conv_tasnet_avse.py — C7: Conv-TasNet-style time-domain AVSE (the headline new candidate).

Why this architecture (see ../../../analysis/results/candidate_scoring.md, CHARTER):
The reference U-Net does not fit because its multi-resolution skip topology forces skip0..3 +
bottleneck to co-reside (2.4 MB peak-live). Conv-TasNet works at a **single latent resolution** with
**no U-Net skips at all**, so that residency wall simply does not exist — while staying in the time
domain (decision D-2). It is also a proven, strong speech-separation/enhancement backbone.

Pipeline:
    audio  [B,1,T]  --encoder(Conv1d stride L/2)-->  w  [B,N,T_lat]
    video  [B,Tv,H,W] --(reused LightweightVideoEncoder)--> [B,Tv,Cv] --align/proj--> v [B,B_ch,T_lat]
    sep:  norm(w) -> 1x1(N->B_ch) -> (+ v) -> TCN(R x X dilated dwsep blocks) -> 1x1(B_ch->N) -> mask
    out:  decoder( w * mask )  ->  [B,1,T']

HW-aware choices: depthwise-separable dilated convs (few DSPs), BatchNorm (foldable), bounded PReLU,
no attention / no dynamic shapes. Quality is the Phase-2 objective; these keep Phase-3 HLS realistic.
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from avse.reference.video_encoder import LightweightVideoEncoder


class _TCNBlock(nn.Module):
    """Conv-TasNet residual block: 1x1 -> PReLU -> BN -> depthwise dilated conv -> PReLU -> BN -> 1x1."""

    def __init__(self, b_ch: int, h_ch: int, kernel: int = 3, dilation: int = 1, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.pad = (kernel - 1) * dilation
        self.in_conv = nn.Conv1d(b_ch, h_ch, 1, bias=False)
        self.prelu1 = nn.PReLU(h_ch)
        self.bn1 = nn.BatchNorm1d(h_ch)
        self.dwconv = nn.Conv1d(h_ch, h_ch, kernel, dilation=dilation, groups=h_ch,
                                padding=0, bias=False)
        self.prelu2 = nn.PReLU(h_ch)
        self.bn2 = nn.BatchNorm1d(h_ch)
        self.out_conv = nn.Conv1d(h_ch, b_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn1(self.prelu1(self.in_conv(x)))
        if self.causal:
            y = F.pad(y, (self.pad, 0))            # causal: pad left only
        else:
            y = F.pad(y, (self.pad // 2, self.pad - self.pad // 2))
        y = self.bn2(self.prelu2(self.dwconv(y)))
        y = self.out_conv(y)
        return x + y                                # residual


class ConvTasNetAVSE(nn.Module):
    """C7 — single-resolution time-domain AVSE mask network with video conditioning.

    Args (Conv-TasNet naming): N encoder filters, L encoder kernel (samples), B bottleneck channels,
    H conv channels, X blocks per repeat, R repeats. Small defaults for fast Phase-2 iteration.
    """

    def __init__(self, config, N: int = 128, L: int = 32, B_ch: int = 64, H_ch: int = 128,
                 X: int = 5, R: int = 2, causal: bool = True):
        super().__init__()
        self.config = config
        self.N, self.L, self.stride = N, L, L // 2

        # ── Audio encoder / decoder (learned filterbank) ──────────────────────────────────────
        self.encoder = nn.Conv1d(1, N, L, stride=self.stride, padding=self.stride, bias=False)
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=self.stride, padding=self.stride, bias=False)

        # ── Video pathway (reuse the validated, cheap encoder) -> bottleneck-channel embedding ──
        self.video_encoder = LightweightVideoEncoder(config)
        v_ch = config.model.video_channels
        self.video_proj = nn.Conv1d(v_ch, B_ch, 1)

        # ── Separator ─────────────────────────────────────────────────────────────────────────
        self.in_norm = nn.BatchNorm1d(N)
        self.bottleneck = nn.Conv1d(N, B_ch, 1, bias=False)
        blocks = []
        for _ in range(R):
            for x in range(X):
                blocks.append(_TCNBlock(B_ch, H_ch, kernel=3, dilation=2 ** x, causal=causal))
        self.tcn = nn.ModuleList(blocks)
        self.mask_conv = nn.Conv1d(B_ch, N, 1, bias=False)

    # -- helpers ----------------------------------------------------------------------------------
    def _video_embed(self, video_frames: torch.Tensor, t_lat: int) -> torch.Tensor:
        # video_frames [B, Tv, H, W] -> [B, Tv, Cv] -> [B, Cv, Tv] -> align to t_lat -> proj to B_ch
        vf = self.video_encoder(video_frames)              # [B, Tv, Cv]
        vf = vf.transpose(1, 2)                            # [B, Cv, Tv]
        vf = F.interpolate(vf, size=t_lat, mode="nearest")  # [B, Cv, t_lat]
        return self.video_proj(vf)                         # [B, B_ch, t_lat]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["mixed_audio"]                           # [B, 1, T]
        T = x.shape[-1]
        w = self.encoder(x)                                # [B, N, T_lat]
        t_lat = w.shape[-1]

        v = self._video_embed(batch["video_frames"], t_lat)  # [B, B_ch, T_lat]

        y = self.bottleneck(self.in_norm(w)) + v           # [B, B_ch, T_lat]
        for blk in self.tcn:
            y = blk(y)
        mask = torch.sigmoid(self.mask_conv(y))            # [B, N, T_lat]

        out = self.decoder(w * mask)                       # [B, 1, ~T]
        if out.shape[-1] >= T:
            out = out[..., :T]
        else:
            out = F.pad(out, (0, T - out.shape[-1]))
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
