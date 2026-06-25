"""avse — Fully on-chip, single-configuration AVSE for RFSoC (ZU48DR).

Package layout:
    avse.data       — LRS3 scene-streaming windowed dataset (Phase-2 training pipeline)
    avse.config     — dataclass config + yaml loader
    avse.metrics    — SI-SDR / PESQ / STOI quality metrics (architecture-agnostic)
    avse.losses     — training losses
    avse.models     — the hardware-shaped architectures (C7 mask, C2 mapping) + reused video encoder

See ../../docs/CHARTER.md for what counts as success.
"""

__version__ = "0.0.1"
