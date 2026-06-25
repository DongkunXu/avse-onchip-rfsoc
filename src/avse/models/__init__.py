"""NEW hardware-shaped AVSE architectures (Phase 2).

Phase-1 picked time-domain candidates (DECISIONS D-2/D-8). Implemented here:
  - C7 ConvTasNetAVSE — single-resolution time-domain mask, no U-Net skips (headline).
  - C2 StreamingTCNAVSE — dilated causal TCN backbone (to come).
The reference U-Net is NOT copied here; see ../reference/ for the teacher (distillation only).
"""

from .conv_tasnet_avse import ConvTasNetAVSE

__all__ = ["ConvTasNetAVSE"]
