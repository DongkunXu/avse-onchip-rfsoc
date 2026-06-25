# analysis/ — Phase 1: analytical working-set screening

**No machine, no training.** The goal is a defensible, quantitative shortlist before spending any GPU
or FPGA time. See [`../docs/ROADMAP.md`](../docs/ROADMAP.md) Phase 1.

## What goes here

| File | Purpose |
|---|---|
| `working_set.py` | The core estimator. Given an architecture spec (per-stage tensor shapes C×T, skip topology, dataflow schedule), compute the **peak simultaneously-live activation working set** (Σ C×T over co-resident tensors) + rough LUT/DSP/BRAM. The binding metric (see CHARTER §3). |
| `validate_baseline.py` | **Mandatory gate.** Encode the reference U-Net (shapes in `docs/reference/bottleneck-diagnosis.md`) and confirm the model reproduces the measured facts: audio_dec ≈ 95 % BRAM standalone, ≈ 215 % BRAM concurrent. If it can't predict the known answer, it can't be trusted on unknowns. |
| `candidates/` | One module per candidate architecture (streaming TCN, recurrent, STFT-mask, tiled-U-Net, DDR-staged-U-Net, single-engine-global-pool, ...). Each defines its tensor/dataflow spec for `working_set.py` to score. |
| `results/` | The output scoring table(s): *approach × peak activation × est. resources × quality risk × engineering effort.* This is the artifact the owner reviews to pick 2–3 directions. |

## Method

1. Build `working_set.py` (the estimator).
2. **Validate it** against the reference design (`validate_baseline.py`) — this must pass first.
3. Specify each candidate under `candidates/`.
4. Generate the scoring table into `results/`.
5. Kill anything that can't fit; cut 6–8 → 2–3.
6. **Owner gate**: review the table, pick directions for Phase 2.

> Discipline (CHARTER §5): score candidates by what makes a *whole* AVSE fit on-chip in one static
> config — not by "how do I shrink the existing U-Net."
