"""Training losses (migrated reference starting point).

`ImprovedAVSELoss` combines time-domain (L1/L2/multi-scale), multi-resolution STFT, and perceptual
(SI-SDR / STOI / PESQ) terms. Depends on `asteroid` + `torch_pesq`. The new architecture may replace
this loss — treat it as a baseline, not a fixture.
"""

from .losses import ImprovedAVSELoss, MultiScaleTimeLoss, MultiResolutionSTFTLoss

__all__ = ["ImprovedAVSELoss", "MultiScaleTimeLoss", "MultiResolutionSTFTLoss"]
