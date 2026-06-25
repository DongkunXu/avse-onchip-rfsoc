"""streaming_tcn_avse.py — C2: streaming dilated-TCN time-domain AVSE.

Same single-resolution, no-U-Net-skips spine as C7, but a deliberately different *architecture
choice* so the two give distinct Pareto points and a clean ablation:

  - C7 (Conv-TasNet): predicts a multiplicative MASK on the encoded mixture (decoder(w * mask)).
  - C2 (this):        DIRECT MAPPING — the TCN predicts the enhanced latent itself (decoder(y_hat)),
                      fully CAUSAL throughout (streaming emphasis, O(state) bounded receptive field).

The bounded receptive field = R x (2^X - 1) latent frames is the "carry small state, peak decoupled
from window length" property from the candidate analysis. HW blocks shared via `_tcn_common`.
"""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._tcn_common import build_tcn, VideoConditioner, tcn_deployable_working_set_elems


class StreamingTCNAVSE(nn.Module):
    """C2 — causal dilated-TCN AVSE with direct (mask-free) latent mapping + video conditioning."""

    def __init__(self, config, N: int = 128, L: int = 32, B_ch: int = 64, H_ch: int = 128,
                 X: int = 6, R: int = 2, causal: bool = True):
        super().__init__()
        self.config = config
        self.N, self.L, self.stride = N, L, L // 2
        self.B_ch, self.H_ch, self.X, self.R = B_ch, H_ch, X, R

        self.encoder = nn.Conv1d(1, N, L, stride=self.stride, padding=self.stride, bias=False)
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=self.stride, padding=self.stride, bias=False)

        self.video = VideoConditioner(config, out_ch=B_ch)

        self.in_norm = nn.BatchNorm1d(N)
        self.bottleneck = nn.Conv1d(N, B_ch, 1, bias=False)
        self.tcn = build_tcn(B_ch, H_ch, X, R, causal=causal)
        self.out_conv = nn.Conv1d(B_ch, N, 1, bias=False)   # direct mapping (no mask multiply)

        # receptive field in latent frames (informational; the "bounded state" of the candidate model)
        self.receptive_field = R * (2 ** X - 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["mixed_audio"]                           # [B, 1, T]
        T = x.shape[-1]
        w = self.encoder(x)                                # [B, N, T_lat]
        v = self.video(batch["video_frames"], w.shape[-1])

        y = self.bottleneck(self.in_norm(w)) + v
        for blk in self.tcn:
            y = blk(y)
        y_hat = self.out_conv(y)                           # enhanced latent (direct mapping)

        out = self.decoder(y_hat)                          # [B, 1, ~T]
        return out[..., :T] if out.shape[-1] >= T else F.pad(out, (0, T - out.shape[-1]))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def deployable_working_set_elems(self) -> int:
        """Bounded streaming activation state (Phase-1-consistent; audio path)."""
        return tcn_deployable_working_set_elems(self.N, self.L, self.B_ch, self.H_ch, self.X, self.R)
