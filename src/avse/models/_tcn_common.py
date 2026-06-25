"""_tcn_common.py — shared building blocks for the time-domain TCN-based AVSE candidates (C7, C2).

Factored out so the two candidates differ only in their *architecture choice* (mask vs direct
mapping, causal depth), not in boilerplate. All blocks are HW-aware: depthwise-separable dilated
convs, BatchNorm (foldable at deploy), bounded PReLU, no attention / no dynamic shapes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from avse.reference.video_encoder import LightweightVideoEncoder


class TCNBlock(nn.Module):
    """Conv-TasNet residual block: 1x1 -> PReLU -> BN -> depthwise dilated conv -> PReLU -> BN -> 1x1.

    causal=True pads on the left only (streaming-friendly); causal=False pads symmetrically.
    """

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
            y = F.pad(y, (self.pad, 0))
        else:
            y = F.pad(y, (self.pad // 2, self.pad - self.pad // 2))
        y = self.bn2(self.prelu2(self.dwconv(y)))
        y = self.out_conv(y)
        return x + y


def build_tcn(b_ch: int, h_ch: int, X: int, R: int, causal: bool = True) -> nn.ModuleList:
    """R repeats of X dilated blocks (dilation 1,2,4,...,2^(X-1))."""
    blocks = []
    for _ in range(R):
        for x in range(X):
            blocks.append(TCNBlock(b_ch, h_ch, kernel=3, dilation=2 ** x, causal=causal))
    return nn.ModuleList(blocks)


def tcn_deployable_working_set_elems(N: int, L: int, B_ch: int, H_ch: int, X: int, R: int,
                                     k: int = 3) -> int:
    """Bounded streaming activation state for a single-resolution causal TCN (Phase-1-consistent).

    These architectures process frame-synchronously, so the on-chip peak is NOT the full-window
    PyTorch activation but the carried state:
      - dilated dwconv ring buffers:  B_ch * (k-1) * sum(dilations) = B_ch*(k-1)*R*(2^X - 1)
      - one latent column in flight:  encoded(N) + bottleneck(B_ch) + tcn working(H_ch) + mask/out(N)
      - raw input ring:               L samples
    Audio path only; the shared video embedding (~30x96) is cheap and common to all candidates.
    """
    history = B_ch * (k - 1) * R * (2 ** X - 1)
    frame = N + B_ch + H_ch + N
    return history + frame + L


class VideoConditioner(nn.Module):
    """Reused (validated, cheap) video encoder -> [B, out_ch, t_lat] conditioning signal."""

    def __init__(self, config, out_ch: int):
        super().__init__()
        self.video_encoder = LightweightVideoEncoder(config)
        self.proj = nn.Conv1d(config.model.video_channels, out_ch, 1)

    def forward(self, video_frames: torch.Tensor, t_lat: int) -> torch.Tensor:
        vf = self.video_encoder(video_frames)          # [B, Tv, Cv]
        vf = vf.transpose(1, 2)                        # [B, Cv, Tv]
        vf = F.interpolate(vf, size=t_lat, mode="nearest")
        return self.proj(vf)                           # [B, out_ch, t_lat]
