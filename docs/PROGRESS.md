# PROGRESS LOG

Running, dated log of what is **done / in progress / next**. Newest entry on top.
One entry per working session or milestone. Keep it factual; rationale goes in
[`DECISIONS.md`](DECISIONS.md), results go in [`../experiments/REGISTRY.md`](../experiments/REGISTRY.md).

Status legend: ✅ done · 🔄 in progress · ⏭ next · ⛔ blocked

---

## 2026-06-25 — Project bootstrap

- ✅ New independent repository created at `G:\phD_Projects\AVSE-OnChip-RFSoC` (own git, parallel to
  the reference deployment; old project untouched).
- ✅ Directory skeleton for the 3-phase funnel (`analysis/ src/avse/ experiments/ hls/ hw/`).
- ✅ Documentation system: `CHARTER`, `ROADMAP`, this `PROGRESS`, `DECISIONS`, and self-contained
  `reference/` facts (bottleneck diagnosis, dead-ends, hardware budget, prior wins, reusable assets).
- ✅ Experiment tracking scaffold (`experiments/REGISTRY.md`).
- ✅ Reusable assets migrated from the reference project (see
  [`reference/reusable-assets.md`](reference/reusable-assets.md)): data pipeline, metrics, loss,
  teacher-model reference, config template.
- 🔄 Virtual environment (`.venv`, Python 3.11, torch **cu128** for Blackwell RTX 5070 Ti) —
  installing via `tools/setup_venv.sh`.
- ⏭ **Phase 1 kickoff**: build `analysis/working_set.py` and validate it against the reference
  design's known 215 % / 95 % BRAM numbers.

### Open items
- ⛔ Teacher checkpoint (`final_model.ckpt`) not yet located on this machine — needed only if Phase 2
  pursues distillation.
- ◇ Owner decision pending before Phase 2: **time-domain vs STFT** (see `DECISIONS.md` D-2).
