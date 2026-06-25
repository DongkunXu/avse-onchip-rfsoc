# PROGRESS LOG

Running, dated log of what is **done / in progress / next**. Newest entry on top.
One entry per working session or milestone. Keep it factual; rationale goes in
[`DECISIONS.md`](DECISIONS.md), results go in [`../experiments/REGISTRY.md`](../experiments/REGISTRY.md).

Status legend: ✅ done · 🔄 in progress · ⏭ next · ⛔ blocked

---

## 2026-06-25 — Phase 3: C7 FITS in real synthesis ✅ (central hypothesis confirmed)

- ✅ Implemented C7's audio mask network in synthesizable HLS (int16; encoder, 10 dilated dwsep TCN
  blocks, hardsigmoid mask, ConvTranspose decoder; placeholder weights — fit is structure-driven, D-9).
- ✅ Vitis HLS 2022.2 csynth (xczu48dr): **C7 audio = 40% BRAM, 41% LUT, 12% DSP** (`hls/RESULTS_csynth_c7.md`).
  - Reference audio path was **152% BRAM** (enc 57 + dec 95) — the reason for the 4-bitstream split.
    C7 brings it to **40%** (~3.8× less) and kills the LUT wall too.
  - + known video (~38% BRAM / ~30% LUT) → whole system ≈ **78% BRAM / ~71% LUT → FITS single static
    config.** The reference's 215% BRAM could not. **Central hypothesis confirmed in real synthesis.**
- ✅ Fixed a tcl root-cause (open_project takes a name, not a path).
- ✅ **Vivado place-and-route (real post-route): C7 audio = 17% LUT, 46% BRAM, 9% DSP, timing MET at
  200 MHz** (LUT far below the csynth 41% estimate). + video → system ≈ **84% BRAM / 47% LUT → fits
  single static config, timing-closed.** Fit confirmed by BOTH synthesis and place-and-route.
- ◇ **OWNER GATE (per D-9 "then come back")**: fit is proven on real hardware reports. Next is the
  high-quality retrain of C7 (full data) + real-weight export. → returning to owner.

## 2026-06-25 — Phase 2: first Pareto ◇ (owner gate — pick the operating point)

- ✅ Implemented C2 (streaming TCN, direct mapping) + shared `_tcn_common` refactor; training harness
  `avse/train.py`; Pareto builder. Fixed warnings + a `.gitignore` inline-comment bug + a pystoi
  sentinel — all at root, no cruft.
- ✅ **Trained C7 and C2** (10 epochs, 10k train windows, dev-160 eval) on the RTX 5070 Ti:
  - **C7 Conv-TasNet (mask): SI-SDR +3.79 dB, PESQ 1.565, STOI 0.690** — still rising at ep9, already
    ≈ the reference FP32 (+3.99 dB) at **0.017 MB** deployable working set (vs reference 4.1 MB).
  - C2 streaming-TCN (mapping): +1.12 dB — dominated by C7 → **masking beats direct mapping** (clean
    ablation, consistent with Conv-TasNet literature).
- ✅ `experiments/build_pareto.py` → `pareto.md` + `pareto.png`. C7 is the standout: near-reference
  quality, smallest working set, fits single-config, strongest circuits story.
- ◇ **OWNER GATE**: pick the Phase-3 operating point. Recommend **C7** as primary (push its training
  further), **C4 tiled** as the high-quality safe anchor. → returning to owner.

## 2026-06-25 — Phase 2 underway: data verified + C7 implemented

- ✅ `tools/verify_data.py`: LRS3 pipeline loads end-to-end on this machine (dev 3365 scenes →
  25272 windows in 13.3s; shapes correct). De-risks training.
- ✅ `experiments/PHASE2_PLAN.md`: C4 anchored to reference quality; C2 + C7 trained; working-set
  instrumentation ties trained quality back to the Phase-1 axis.
- ✅ `src/avse/models/conv_tasnet_avse.py` (**C7**): Conv-TasNet-style time-domain AVSE, no U-Net
  skips, reuses the validated video encoder, HW-aware (dwsep dilated convs, BN, PReLU). **308,544
  params**; forward+backward verified on CPU and on the RTX 5070 Ti (Blackwell). `onchip_config.yaml`
  given a minimal video-only `model:` section (audio backbone params are per-candidate Python args).
- ⏭ **Next**: implement C2 (streaming TCN), build the training harness + working-set instrumentation,
  run short trainings to validate the loop, then scale to the quality-vs-working-set Pareto.

## 2026-06-25 — Owner gate resolved; candidate set expanded (D-2, D-8)

- ✅ Owner decisions recorded: **D-2 = time-domain only** (C3/STFT parked), **D-8 = prototype C4 + C2 +
  C5(plan-B)** with an explicit mandate to explore beyond the old docs.
- ✅ Self-derived new time-domain candidates added & scored: **C7 Conv-TasNet-style mask** (no
  multi-resolution skips → no residency wall; sys 67%, fits — the headline new candidate), plus
  levers **C8 recompute-skip** (sys 94%, borderline alone) and **C9 compressed-skip** (sys 103%, must
  combine). C3 marked out-of-scope. (`results/candidate_scoring.md` regenerated.)
- ✅ **Phase-2 set decided: C4 (anchor/control) + C2 (TCN) + C7 (Conv-TasNet, root-cause removal)**;
  C5 plan-B; C8/C9 combinable levers.
- ⏭ **Next (Phase 2)**: plan + verify the LRS3 data pipeline end-to-end, then implement the 3
  architectures and train small versions for the quality-vs-working-set Pareto.

## 2026-06-25 — Phase 1: candidate scoring table produced ◇ (owner gate)

- ✅ `analysis/candidates.py` — 7 candidates encoded (C0 reference, C6 pool-only, C5 DDR-staged,
  C4 tiled, C1 streaming-chunked, C2 streaming-TCN, C3 STFT-mask), each scored by the **validated**
  peak-live model. Fit verdict is on the whole system (audio peak-live + ~65% shared video/fusion/
  weight overhead).
- ✅ `analysis/score_candidates.py` → `results/candidate_scoring.md`. Result:
  - **C0 / C6 DO NOT FIT (~140%)** — sharpened proof that Axis-3 (pool/schedule the same U-Net) alone fails.
  - **C4 tiled (74%), C1 chunked (70%), C2 TCN (77%), C3 STFT (67%) all FIT**; C5 DDR-staged borderline (90%).
- ✅ Fixed two issues honestly: a `%`-format bug (literal `%` in a `%`-formatted string → f-string),
  and a correctness gap (fit verdict now counts the shared video/fusion overhead, not audio-only).
- ◇ **OWNER GATE**: pick the 2–3 directions for Phase 2 + resolve DECISIONS **D-2 (time vs STFT)**,
  which gates whether C3 is in scope. Recommended shortlist: **C4 (safe anchor) + C2 (novelty, time) +
  C3 (highest novelty, needs D-2)**. → returning to owner.

## 2026-06-25 — Phase 1: working-set model built & VALIDATED ✅

- ✅ Investigated the reference HLS source (`UNet-AVSE-Vitis/src/ip_*/...top.cpp`, `common/types.hpp`)
  to get ground-truth buffer lists + partition pragmas (not guessed). data_t = ap_fixed<16,7> (16-bit).
- ✅ `analysis/working_set.py` — core model: `Buffer/Module/Design`, first-principles BRAM18/URAM
  mapping (RAMB18=1024 words, cyclic-partition banking, ping-pong), + liveness `peak_live_working_set`.
- ✅ `analysis/baseline_reference.py` — the reference 4-IP design encoded from the real buffer lists.
- ✅ `analysis/validate_baseline.py` — the trust gate. **PASSES**: audio_dec (binding IP) predicted
  0.96× measured (91.2 % vs 95 %), URAM exact (36.2 % vs 36 %), concurrent activation 193 % vs measured
  215 % total (gap = known weight+working residual). Activations confirmed as the wall, quantitatively.
- ✅ Fixed a real bug honestly (not bypassed): cp1252 console crashed on a Unicode glyph →
  root-caused to stdout encoding; reconfigured stdout to UTF-8 + ASCII-only report symbols.
- ✅ Key finding (`analysis/results/baseline_validation.md`): audio static 4.1 MB vs **peak-live 2.4 MB
  (1.70×)** — even perfect pooling of the same U-Net topology can't fit; a winner **must bound the
  live temporal extent** (Axis 1/2), not just schedule (Axis 3). First model-backed steer.
- ⏭ **Next**: encode candidate architectures as `Design`s under `analysis/candidates/` and score them
  with the validated model → the *approach × peak-activation × resources × risk × effort* table.

### Notes / known approximations
- Video IP modelled to 0.63× (DATAFLOW ping-pong + dense temporal weights inflate it); deliberately
  not over-fit since video is not the residency wall. Documented in the results file.

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
