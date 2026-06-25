# PROGRESS LOG

Running, dated log of what is **done / in progress / next**. Newest entry on top.
One entry per working session or milestone. Keep it factual; rationale goes in
[`DECISIONS.md`](DECISIONS.md), results go in [`../experiments/REGISTRY.md`](../experiments/REGISTRY.md).

Status legend: ✅ done · 🔄 in progress · ⏭ next · ⛔ blocked

---

## 2026-06-26 — Scene-pool memory fix: bound by BYTES, not scene count ✅

- ⛔→✅ The first full-data run on the new pipeline finished **epoch 0 (SI-SDR +2.21 dB)** then **crashed at
  the epoch-0→1 transition** with `MemoryError` in a DataLoader worker (`np.load` of a 16.3 MB scene npy).
  Root cause (DECISIONS **D-16**): the sliding scene pool was capped by **scene COUNT** (`scene_buffer=128`),
  but LRS3 scene lengths are **long-tailed** — a 74 s scene is ~25 MB resident (16 MB video + ~9 MB audio).
  When epoch 1's shuffle clustered many long scenes into the pool, 128 scenes × 6 workers blew past the
  32 GB host's commit limit. Counting scenes was the wrong invariant.
- ✅ **Fix (structural):** the pool is now bounded by a **per-worker byte budget** (`--pool-mb`, default
  160 MB), with `--scene-buffer` as a secondary count cap; `fill()` stops at whichever binds first and
  always keeps ≥1 scene resident. Resident pool ≤ budget + one scene, *independent of the scene-size
  distribution*. Both knobs are CLI args so RAM can be tuned without code edits.
- ✅ **Verified two ways:** (1) single process streaming the **whole** shuffled train set (incl. the longest
  scenes), 30 k windows — peak RSS **823 MB**, flat, no OOM; (2) 6-worker run **crossed the ep0→1 boundary**
  (ep0 complete, ep1 to 70%) with no MemoryError. `--quick` smoke unchanged (numerics identical).
- ⏭ Owner relaunches `p2-c7-full` (command unchanged; add `--pool-mb 120` if RAM is ever tight).

## 2026-06-25 — Dead-code cleanup (migrated teacher + Lightning removed) ✅

- ✅ Removed all unused migrated-from-reference code (DECISIONS **D-15**), done **without disturbing the
  live full-data training** (deleted files were not in the running process's import graph; verified in an
  isolated CPU process — both models import + build, C7 still 308,544 params):
  - deleted the dead 0.37 M teacher: `src/avse/reference/{audio_encoder,avse_model,fusion,__init__}.py`
    (distillation was never pursued — C7 trained from scratch);
  - deleted `src/avse/data/data_module.py` (PyTorch-Lightning DataModule — `train.py` builds its own loader);
  - deleted `src/avse/config/reference_base_config.yaml` (old U-Net config, referenced only in a comment);
  - **promoted** the one reused piece, `LightweightVideoEncoder`, from `avse/reference/video_encoder.py`
    to `avse/models/video_encoder.py` (it is a native component now, not "reference"); fixed the import in
    `_tcn_common.py`; dropped the now-unused `pytorch-lightning` dep; refreshed package docstrings.
  - kept (NOT junk): `analysis/` (Phase-1 validated working-set model) and `docs/reference/` (the
    intentionally self-contained inherited facts). Also cleared stray logs / throwaway quick-run dirs.

## 2026-06-25 — Data pipeline I/O refactor: scene-streaming kills the disk wall ✅

**Symptom the owner saw:** GPU utilization periodically collapsing to 0% while the dataset SSD (D:) spiked.
Sustained hardware sampling (77 samples) confirmed it was not occasional — **D: was pegged the whole run**
(disktime mean 437%, queue ~4.4, steady ~40 MB/s) while the **GPU was starved 51% of the time** (util mean
45.9%), CPU only 7.5% → pure I/O wait, not compute.

- ✅ **Root cause (structural, not a tuning issue):** the map-style `AVSEDataset.__getitem__` decodes one
  *window* by opening that window's 3 files from scratch. With the DataLoader's global shuffle, consecutive
  windows come from different scenes, so with **~315k windows over ~34.5k scenes (~9 windows/scene)** each
  scene's `.wav`/`.wav`/`.npy` got re-opened **~9× per epoch** and re-resampled ~9×. That random-small-read
  pattern caps an NVMe SSD at ~40 MB/s (latency/IOPS-bound: only ~80 window-reads/s) and starves the GPU.
- ✅ **Pre-packing ruled out by disk facts (checked, not assumed):** train = **260 GB** (scenes 173.7 +
  video 86.8), **D: has only 70 GB free** → a repacked copy doesn't fit. Fix had to be access-pattern, zero
  extra disk.
- ✅ **Fix = make the unit of work a SCENE, not a window** (`src/avse/data/stream_dataset.py`,
  `AVSESceneStreamDataset(IterableDataset)`): each worker takes a disjoint scene shard; each scene's 3 files
  are opened **once per epoch**, read whole sequentially into RAM, resampled once, then all ~9 windows are
  sliced from memory. SGD shuffling preserved by a bounded **sliding scene pool** (≈128 scenes resident per
  worker, windows drawn at random from the pool → a scene's overlapping windows spread across batches);
  only raw arrays are buffered (decoded tensors live one-at-a-time) so memory stays ~1.6 GB across 6 workers.
  Per-window slice + the joint `0.8/max` normalization are **byte-identical** to the old path → training
  numerics unchanged, only I/O differs. `train.py` switched to it; resume/checkpoint untouched; old
  `AVSEDataset` kept intact for `verify_data.py` / val-compat.
- ✅ **Verified by A/B on real load** (train subset, workers 6 / batch 48, 60 samples):
  GPU util **45.9% → 73.7%** (steady-state 80–100%), GPU-starved **51% → 20%**, D: disktime **437% → 26%**,
  D: queue **4.37 → 0.26**. The disk wall is gone; training is now GPU-bound. `--quick` smoke passes
  end-to-end (loss falls, eval + checkpoints write).
- ⏭ **Next:** relaunch the definitive full-data run `p2-c7-full` on the optimized pipeline (now ~1.6× faster
  wall-clock and rising headroom), then export `best.pt` into the HLS ROMs.

## 2026-06-25 — Full-data run prepared & hardened; ONE open task

**Where we are: Phases 1–3 essentially done (C7 chosen, fit confirmed on real synth+P&R). The single
open task is the owner-launched full-data C7 quality run, then real-weight export into the HLS ROMs.**

- ✅ Training harness made production-grade for the long run (`src/avse/train.py`):
  - **Dual ASCII progress bars** (epochs/batches, single-line, live SI-SDR/PESQ/STOI/best/patience),
    no emoji/CJK; **early-stop** (patience 5); **80 epochs**; full data; **per-epoch resumable
    checkpoint** + `--resume`; per-epoch **trend.png**.
  - **On-disk window cache** (`.dataset_cache/`, git-ignored): the ~13–23 min train scan (34.5k scenes
    → 315 253 windows) now loads in **~0.2 s** on rerun/resume. ASCII scan progress bar shown (was
    fully suppressed → looked hung). Train cache pre-built and verified.
  - **NaN guard**: some LRS3 windows have an all-zero (silent) target → torch_pesq PESQ loss = NaN →
    poisoned weights. Now **non-finite-loss batches are skipped** (no backward/step); `nan-skip N`
    reported per epoch. Confirmed: a silent batch is skipped, weights stay finite. (This bit the first
    full-run attempt at ep0 — fixed before the real run.)
- ⏭ **OPEN TASK (owner launches)**: the full-data run, then export `best.pt` weights into the HLS
  weight ROMs (`hls/src/*`, currently placeholder) for a quality-accurate deployment + final eval.
  ```
  cd G:\phD_Projects\AVSE-OnChip-RFSoC
  .\.venv\Scripts\python.exe -m avse.train --model c7 --exp-id p2-c7-full --epochs 80 \
      --early-stop-patience 5 --batch 32 --workers 4 --prefetch 4 --max-train-windows 0 --lr 5e-4
  ```
  (Resume with the same line + `--resume`. Do NOT run it while Vivado P&R runs — 32 GB host, D-11.)

## 2026-06-25 — C7 40k high-quality run done; full-data run ready

- ✅ `p2-c7-hq` (C7, 40k-window subset, 20 epochs, cosine LR) finished: **best SI-SDR +4.89 dB (ep16)**,
  PESQ 1.683, STOI 0.718; final ep19 +4.79. **Exceeds the FP32 reference (+3.99 dB)** at **0.017 MB**
  working set (1/240 of the reference). Trajectory 2.82 → 4.25(ep5) → 4.69(ep10) → 4.89(ep16). Pareto
  regenerated (now uses best-SI-SDR epoch). C7 clearly has headroom; full data expected to push higher.
- ✅ Training harness upgraded for the definitive run: dual ASCII progress bars (epochs/batches, live
  metrics, single-line), early-stop (patience 5), 80 epochs, full data, per-epoch resumable checkpoint
  + trend.png. CPU-verified (smoke + resume). Owner will launch the full-data run.
- ⏭ **Owner to launch** the full-data run (all 315k windows, 80 epochs, early-stop 5) with the
  configured command (see chat / PHASE2_PLAN). Then export best.pt real weights into the HLS ROMs for
  a quality-accurate Phase-3 deployment.

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
- ✅ Fit proven on real hardware reports (synth + P&R). Owner gate resolved (**D-10**):
  next = (1) **monolithic integrated synth** (C7 audio + video in ONE design → real total fit),
  then (2) **high-quality retrain** on full data. Parallelised: retrain on GPU, monolithic synth on CPU.
- ✅ **Monolithic integrated synth** (C7 audio + video in ONE design, `c7_avse_top`): whole-system
  csynth = **70% BRAM, 90% LUT (estimate), 28% DSP — fits one static config** (`hls/RESULTS_avse_monolithic.md`).
  vs reference 215% BRAM / 126% LUT (4 bitstreams). LUT 90% is the csynth over-estimate (standalone
  went 41%→17% post-route); Vivado P&R running for the definitive total.
  - Clean refactor: audio compute extracted to `c7_audio_core.hpp` (shared by both tops).
- ✅ **MONOLITHIC P&R DONE — the definitive whole-system total**: complete AVSE (C7 audio + video,
  ONE design) post-route = **80% BRAM, 41% LUT, 20% DSP, 200 MHz timing MET** (`hls/RESULTS_avse_monolithic.md`).
  vs reference 215% BRAM / 126% LUT (4 PCAP bitstreams). **The whole AVSE fits in ONE static config,
  timing-closed — the project's central goal, proven on real place-and-route.**
  - Hit a snag first: ran the monolithic P&R concurrently with the GPU retrain on this 32 GB host →
    memory contention hung Vivado (2 h, 0% CPU) and crashed the retrain. Killed the stuck procs,
    serialized the jobs, re-ran the P&R alone (fresh project) → clean. Lesson: **never run Vivado P&R
    and GPU training concurrently on this 32 GB machine** (see DECISIONS).
- 🔄 High-quality retrain `p2-c7-hq` RUNNING ALONE (fixed harness): C7, **40k-window subset** of the
  315k-window train split, **data_mode=full** (noise + speech interferers), 20 epochs cosine LR.
  Healthy: ep2 already +3.83 dB SI-SDR (> the 10-epoch run) and rising; ~437 s/epoch; VRAM 4/16 GB,
  no spill. ETA ~20:30.
- ⏭ Planned (owner, see `experiments/PHASE2_PLAN.md`): after this, a **full-data run** (all 315k
  windows) + **hardware tuning** to squeeze the box (raise batch toward the 16 GB VRAM ceiling with a
  ~1–2 GB margin, never spill to shared; more workers/prefetch; sweep batch/LR) → definitive quality +
  real-weight export.

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
