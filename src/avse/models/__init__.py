"""NEW hardware-shaped AVSE architectures (Phase 2).

Phase-1 picked time-domain candidates (DECISIONS D-2/D-8). Implemented here:
  - C7 ConvTasNetAVSE  — single-resolution time-domain MASK, no U-Net skips (headline).
  - C2 StreamingTCNAVSE — causal dilated-TCN, direct (mask-free) MAPPING.
Shared HW blocks live in _tcn_common; the (validated, reused) video pathway is video_encoder.py.
"""

from .conv_tasnet_avse import ConvTasNetAVSE
from .streaming_tcn_avse import StreamingTCNAVSE

__all__ = ["ConvTasNetAVSE", "StreamingTCNAVSE"]
