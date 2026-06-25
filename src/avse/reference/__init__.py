"""The 0.37 M reference ("teacher") AVSE model — FOR DISTILLATION / COMPARISON ONLY.

⚠️ This is the OLD time-domain U-Net whose 4-bitstream form factor this project exists to escape
(see ../../../docs/CHARTER.md and docs/reference/bottleneck-diagnosis.md). It is kept here ONLY as a
distillation teacher / quality yardstick. **Do not make it the design center of the new architecture.**

Migrated verbatim. Known caveat: `avse_model.py` has `from utils.losses import ImprovedAVSELoss`
(the old project layout) — in this project the loss lives at `avse.losses`. The sub-encoders
(`audio_encoder`, `video_encoder`, `fusion`) import cleanly; only the top LightningModule's loss
import needs fixing if/when the full teacher is instantiated. See docs/reference/reusable-assets.md.
"""
