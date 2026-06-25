"""avse — Fully on-chip, single-configuration AVSE for RFSoC (ZU48DR).

Package layout:
    avse.data       — LRS3 windowed dataset + Lightning data module (migrated, reusable)
    avse.config     — dataclass config + yaml loader (migrated)
    avse.metrics    — SI-SDR / PESQ / STOI quality metrics (clean, architecture-agnostic)
    avse.losses     — training losses (migrated reference; may be replaced per architecture)
    avse.models     — NEW hardware-shaped architectures (written fresh in Phase 2)
    avse.reference  — the 0.37 M teacher model defs (distillation / comparison ONLY)

See ../../docs/CHARTER.md for what counts as success.
"""

__version__ = "0.0.1"
