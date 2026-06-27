# PROGRESS LOG

Running, dated log of what is **done / in progress / next**. Newest entry on top.
One entry per working session or milestone. Keep it factual; rationale goes in
[`DECISIONS.md`](DECISIONS.md), results go in [`../experiments/REGISTRY.md`](../experiments/REGISTRY.md).

Status legend: ✅ done · 🔄 in progress · ⏭ next · ⛔ blocked

---

## 2026-06-27 — B3 real-weight synthesis: diagnosed a 6 h HLS blow-up; audio+video synth FAST + FIT 🔄

**The first real-weight monolithic csynth ran ~6 h without producing a report and was lost when the session
ended.** Investigated honestly (not re-run blindly):
- ✅ **It produced no result** — no `c7_avse` report anywhere; it spent ~16 min in Standard Transforms then
  the remaining ~5.7 h in scheduling/binding and never reached report generation. (Owned a mistake: relaunched
  with `open_project -reset` which overwrote the old detailed log — but the old run had no report to lose.)
- ✅ **Isolated the cause** by synthesizing each core standalone (real weights): the **audio core synthesizes
  in ~3 min and FITS** — **BRAM 1031 (47%), DSP 601 (14%), LUT 122566 (28%), ~200 MHz** (DSP up from the
  placeholder ~12% because the `bn_t<32,16>` inline in_norm/bn affines are wide multiplies; still comfortable).
  The 6 h was **entirely the faithful video encoder**.
- ✅ **Root cause (video):** over-aggressive per-loop pipelining (II=2) forced wide (×96) reduction unrolls +
  bank-conflict analysis on **strided shortcut reads into cyclic-partitioned buffers**, and complete-
  partitioned the `v_fp_w`(36864)/`v_tp_w`(9216) weight ROMs into registers → HLS scheduling/binding exploded.
- ✅ **Fix (DECISIONS D-19):** the video is NOT the throughput bottleneck (audio dominates latency), so it is
  synthesized in a **conservative ROLLED schedule** (no per-loop pipeline / no buffer partition) — functionally
  identical (C-sim still valid), low resource, synthesizes in **~5 min**: standalone video (rolled, real
  weights) = **BRAM 489 (22%), DSP 226 (5%), LUT 69391 (16%)**, latency 2.2 s (slow — that is the deferred
  throughput-optimization target, owner-directed). Diagnostic standalone top: `c7_video_top.cpp`.
- ✅ **Monolithic `c7_avse_top` csynth DONE (real weights) — WHOLE AVSE FITS one static config:**
  **BRAM 1603 (74%), DSP 833 (19%), LUT 189313 (44%)**, ~9 min synth. (First pass was 87% BRAM; a trivial
  over-pipelined `VPROJ` had auto-complete-partitioned `video_feat` into 96 banks for +13% BRAM — rolled it
  too, fit-first, D-19.) audio_core 47% BRAM / 14% DSP / 27% LUT (pipelined); video_encoder 22% / 5% / 15%
  (rolled). Latency 2.53 s (rolled video dominates — the deferred throughput-optimization target). Comparable
  to the placeholder csynth (70% BRAM) but now with the REAL trained weights and C-sim-validated computation.
- ✅ **B4 DONE — Vivado P&R of the real-weight monolithic: FITS one config, timing MET @ 200 MHz.**
  `export_design -flow impl` (~33 min, not the feared 2.5 h — the rolled video places fast). **Post-route:
  BRAM 1843 (85%), LUT 80933 (19%), DSP 720 (17%), FF 41711 (5%); CP 4.869 ns < 5.0 → 200 MHz, WNS
  +0.131 ns.** Packaged IP `c7_avse/sol1/impl/export.zip`. (Note: the Start-Process wrapper reported "exit 1"
  but vitis_hls exited cleanly — verified via the impl log "Timing met" + export.zip; a false alarm.)
  vs placeholder post-route (80% BRAM / 41% LUT): real weight ROMs raise BRAM to 85%, rolled video drops
  LUT to 19%. **The whole real-weight, C-sim-validated AVSE fits ONE static config, timing-closed.** Full
  numbers in `hls/RESULTS_avse_monolithic.md`.
- 🔄 **On-board flow STARTED (owner: board is PYNQ-ready, I drive it):**
  - ✅ Static single-config **block design built + validated** (`hw/tcl/01_build_bd.tcl`): zynq_ultra_ps_e
    (RFSoC 4x2 board preset) + SmartConnects — PS HPM0 → IP `s_axi_control`; IP's 3× m_axi (gmem0/1/2,
    audio_in/video_in/audio_out) → HP0 → DDR. (Used the reference `UNet-AVSE-Vitis/dfx` static-BD flow as a
    guide — single static config, not its 4-PCAP DFX.) License: `XILINXD_LICENSE_FILE` →
    `G:/phD_Projects/LICENSE_FOR_ISE_VIVADO.lic`.
  - 🔄 Bitstream build RUNNING (`hw/tcl/02_build_bitstream.tcl`, detached) → exports PYNQ overlay
    `hw/overlay/avse_sys.{bit,hwh}`.
  - ✅ Board scripts ready + prep validated: `hw/board/run_fpga.py` (board-side PYNQ driver — Overlay,
    allocate, write the 3 buffer phys-addrs to regs 0x10/0x1c/0x28, ap_start→poll ap_done),
    `tools/prep_board_windows.py` (PC: dev windows → int16 matching the emulator; verified ranges
    audio_in ±26214, video_in [0,405]), `tools/score_board.py` (quality + silicon-vs-emulator check).
    Subset run (~2.5 s/window) confirms silicon reproduces the C-sim-validated 4.98 dB output.
- ✅ **First on-board run — single-config bitstream RUNS on real RFSoC 4x2** (PYNQ 3.0.1, aarch64): overlay
  loads in 2.3 s, IP runs, 16 windows complete, ap_done OK, zero-in→zero-out correct. The whole AVSE runs in
  ONE static bitstream on real silicon. Board access via plink/pscp (`-hostkey`), `/home/xilinx/avse_onchip/`.
- ⛔→✅ **Found + fixed a real hardware bug the C-sim could not catch.** Real data gave SI-SDR **−13 dB**
  (worse than mixed) with **periodic corruption every STRIDE=16 samples**; silicon ≠ emulator (rms_diff ≈
  signal). Root-caused to the **decoder scatter-accumulate** (`obuf[s] += ` with a computed index): pipelining
  the overlap-add (consecutive (n,t) hit the same `obuf[s]`) is a read-modify-write hazard that sequential
  C-sim hides but real hardware loses updates on. **Fix: roll the DEC loop** (sequential = exact; hazard-free
  GATHER decoder deferred to the optimization phase). All other loops are write-once gathers (unaffected).
  Re-csynth (rolled decoder): 72% BRAM / 19% DSP / 44% LUT. 🔄 bitstream rebuild running.
- Note: measured on-board compute = **11.67 s/window** (not the 2.5 s csynth estimate) — the rolled video
  reads `video_in` element-by-element from DDR with no bursting; this is the prime throughput-optimization
  target (D-19), to tackle after the corrected end-to-end run is confirmed.
- ⏭ When the rebuilt bitstream is done: redeploy + re-run the 16 windows → confirm silicon ≡ emulator (the
  4.98 dB on real hardware). Then (separate phase) throughput optimization (D-19). See
  [[hls-synthesis-and-optimization]] memory.

## 2026-06-27 — Phase 3b kickoff: real-weight deployment + deployment-accurate quality 🔄

**Next task started: turn the FP32 quality number into the on-chip int16-deployment number, and wire the
real weights into HLS.** The Phase-2 result (5.40/1.727/0.754) is FP32 PyTorch; the owner wants the
*actual* fixed-point (ap_fixed<16,7>) on-chip quality — the number comparable to the reference deployed-INT16
anchor (5.46/1.743/0.738).

- ✅ **Reconciliation spec** (`hls/DEPLOY_PLAN.md`): pinned the exact deploy compute graph (PyTorch op →
  deploy math → HLS array) for the whole AVSE (audio core + video encoder), the fixed-point formats, the BN
  folding formulas, and **every fidelity gap** between the trained model and the current (placeholder, D-9)
  HLS. Method mirrors Phase 1: build a faithful fixed-point *emulator*, validate it against HLS C-sim, run at
  scale. Two decoupled deliverables: **A** = the number (Python emulator, no synth), **B** = the silicon
  (faithful HLS + real ROMs + re-synth).
- Investigation found real gaps the placeholder fit-check masked (fit was structure-driven, so output
  correctness was never checked): **G1** encoder latent is **T_LAT=1201**, not 1200; **G2** the HLS decoder
  has a 16-sample (one-STRIDE) offset vs PyTorch ConvTranspose(pad=16); **G3** the TCN blocks' bn1/bn2 are
  dropped (must fold into dwconv/out_conv biases); **G5–G8** the video encoder is a cost-proxy (no BN, no
  residual shortcuts, global-mean-pool instead of AvgPool+feature_proj); **G9** video proj/temporal biases
  missing. All catalogued in DEPLOY_PLAN with the faithful fix. None hard — all mechanical — but all required
  for "actual performance" to be real.
- ✅ **Deliverable A DONE — the deployment-accurate quality number.**
  - `tools/export_weights.py`: best.pt → fold every foldable BN (video convs; in_norm/bn1/bn2 kept inline to
    respect the zero-pad boundary) → one weight truth source `deploy_weights.npz`. BN-fold unit checks PASS.
    Surfaced a real range fact: `in_norm` scale reaches ~102 (low-variance channels) — would overflow
    `wgt_t<16,5>`; resolved by keeping inline affines high-precision (DECISIONS **D-18**).
  - `tools/c7_fixedpoint.py`: bit-faithful fixed-point emulator (video encoder + audio core) from the npz,
    `precision={fp,int16} × mask={sigmoid,hardsigmoid}`. **Correctness gate PASS**: emulator fp+sigmoid ≡
    PyTorch best.pt (max|Δ|=4.4e-4, ref rms 6.7e-2; each component verified ≤1e-5).
  - `tools/eval_deploy.py`: same protocol as `eval_full_dev`, forward = emulator. **Chain validated**:
    fp+sigmoid full-dev = **5.399/1.727/0.754**, reproduces the committed PyTorch number to 3 decimals.

  | full-dev (3327 scenes) | SI-SDR | PESQ-WB | STOI | Δ vs FP32 |
  |---|---:|---:|---:|---|
  | FP32 (sigmoid) | 5.399 | 1.727 | 0.754 | — |
  | int16 (sigmoid) | 5.069 | 1.634 | 0.746 | quant −0.330 dB |
  | **int16 + hardsigmoid (on-chip deploy)** | **4.984** | **1.632** | **0.742** | **−0.415 dB** total |

  **The on-chip int16 C7 AVSE delivers 4.98 dB SI-SDR / 1.63 PESQ / 0.742 STOI on full dev, single static
  config** — still **beating the FP32 teacher anchor** (3.99/1.673/0.741) on SI-SDR (+0.99) and STOI, at
  1/240 the working set. int16 quantization costs −0.33 dB; the cheap HW **hardsigmoid mask adds only
  −0.085 dB** (good HW story). Below the deployed-INT16 reference (5.46, but N=496 subset).
- 🔄 **Deliverable B (silicon) — B1 DONE: audio core value-faithful + C-sim validated.**
  - `tools/gen_hls_weights.py` → `hls/src/c7_weights.hpp` (real ROMs: `wgt_t` MAC operands, `bn_t`=ap_fixed
    <32,16> for inline BN/in_norm affine & PReLU & folded biases). Added `bn_t` to `c7_types.hpp`; set
    **T_LAT=1201** (G1).
  - Rewrote `c7_audio_core.hpp` value-faithful: real ROMs, inline bn1/bn2 (PReLU in acc_t → one data_t cast,
    matches the emulator), **fixed the decoder offset** `s=t·STRIDE+k−STRIDE` (G2). Fixed the monolithic
    upsample index (G10).
  - `tools/dump_hls_vectors.py` (golden vectors from the emulator) + `hls/tb/tb_audio_core.cpp` +
    `run_csim_audio.tcl`. **Vitis HLS C-sim PASS**: HLS `c7_audio_top` reproduces the emulator to
    **rel_rms 0.3–0.6%, max ~1.5 data_t LSB**; the max-diff samples are interior (i=17228, i=5678), not at
    the window boundaries — confirming G1/G2 are correct and the residual is float-emu-vs-fixed-point
    boundary jitter (≈ ±1 LSB), not a logic bug. So the 4.98 dB emulator number is faithful to silicon to
    within ~0.02 dB.
- ✅ **B2 DONE: faithful video encoder + end-to-end C-sim.** Rewrote `c7_video.hpp` value-faithful (G5–G8:
  conv0 BN fold; DWSep depthwise→pointwise+BN+ReLU **+ residual shortcut [1×1 s2 + BN]**, dropped the proxy's
  extra depthwise ReLU; AvgPool(k5,s1)→2×2 then feature_proj Conv2d(k2)+ReLU; temporal_proj+residual) with
  real ROMs; `c7_avse_top.cpp` now uses the real video proj (Conv1d+bias, G9) and the correct upsample index
  (G10). **End-to-end Vitis HLS C-sim PASS** (`tb_avse` + `vectors_full.txt`): the monolithic `c7_avse_top`
  reproduces the emulator to **rel_rms 0.5–0.85%, max 2–3 data_t LSB** (interior samples) → the whole
  synthesizable AVSE computes the 4.98 dB number within ~0.02 dB. Emulator ≡ silicon confirmed.
- ⏭ **B3:** csynth the monolithic — resource check for the faithful video encoder (the flagged fit risk).
  **B4:** Vivado P&R → the final real-weight fit numbers.

## 2026-06-27 — Full-data run DONE + definitive full-dev evaluation ✅ (quality ≈ reference, fits single-config)

- ✅ **`p2-c7-full` finished** (early-stopped ep23 on val total-loss; best = **ep18**, `best.pt`).
- ✅ **Definitive full-dev evaluation** (`tools/eval_full_dev.py`, mirrors the reference protocol in
  `test reference/scripts/evaluate_snr_bins.py`: per-scene sliding windows → per-window STOI/PESQ-WB/SI-SDR,
  enhanced AND mixed-baseline, scene-weighted over **all 3327 dev scenes**; metric fns byte-identical to the
  reference; training-consistent per-window 0.8/|mixed| norm = the on-chip deployment norm). Efficient:
  GPU forward + 16-proc CPU pool → 5 min for 25k windows × 2 metrics (vs ~34 min single-thread).

  | model / anchor | N scenes | SI-SDR dB | PESQ-WB | STOI | working set | single-config fit |
  |---|---:|---:|---:|---:|---:|:--:|
  | **C7 full-data (p2-c7-full, ep18)** | 3327 | **5.40** | **1.727** | **0.754** | **0.017 MB** | ✅ |
  | C7 hq (p2-c7-hq, 40k subset) | 3327 | 4.32 | 1.673 | 0.729 | 0.017 MB | ✅ |
  | mixed input (baseline) | 3327 | 2.05 | 1.522 | 0.713 | — | — |
  | *Reference FP32 teacher (cited, REGISTRY)* | 3319 | 3.99 | 1.673 | 0.741 | — | ❌ (teacher) |
  | *Deployed INT16 ref (cited, REGISTRY)* | 496 | 5.46 | 1.743 | 0.738 | 4.10 MB | ❌ 215% BRAM |

- ✅ **Result, on the most comparable anchor** (FP32 teacher, **same full-dev scale N≈3300**): C7 full-data
  **beats it on all three** — SI-SDR **+1.41 dB** (5.40 vs 3.99), PESQ **+0.054**, STOI **+0.013** — at
  **1/240 the working set**, single static config. Vs the deployed-INT16 number (5.46/1.743/0.738) C7 is on
  par (SI-SDR/PESQ within rounding, STOI higher), but that anchor is a much smaller **N=496** subset so it is
  indicative, not a controlled head-to-head. Either way the CHARTER's allowance to sacrifice quality
  substantially was not needed. Full-data training added **+1.08 dB / +0.054 PESQ / +0.025 STOI** over the
  40k hq run (same protocol). Abs. improvement over noisy input: **+3.35 dB / +0.20 PESQ / +0.04 STOI**.
- Note: the 200-window training val (~25 scenes) was noisy — it overstated hq (4.89) and understated full
  (ep18 5.07); the full-dev numbers above are the reliable ones. The two reference anchors (3.99 FP32 vs 5.46
  INT16) are NOT inconsistent — they are different eval subsets (N=3319 vs N=496); see REGISTRY. Eval outputs
  are git-ignored (`experiments/**/full_*_eval.json`); reproduce with `tools/eval_full_dev.py --ckpt <best.pt>`.
- ⏭ Next: export `best.pt` real weights into the HLS weight ROMs (`hls/src/*`) for a quality-accurate
  deployment + final end-to-end check.

## 2026-06-26 — Model selection on val TOTAL LOSS (not SI-SDR alone) ✅

- ✅ `best.pt`/early-stop were tracking **val SI-SDR only** — one term (w=0.5) of a 7-term objective that
  also weights PESQ (3.0) and STOI (4.0). Switched selection + early-stop to the **validation total loss**
  (the comprehensive training objective on held-out data), the owner's call (DECISIONS **D-17**).
  `evaluate()` now computes the val total loss too (NaN-guarded, D-12); SI-SDR/PESQ/STOI still logged.
- ✅ **Logic-only — no retraining.** Training/loss/optimizer/data unchanged; only the "best" epoch and
  early-stop trigger change. Resume-safe: the live ep17 checkpoint (old format, `best_sisdr` only) triggers
  a clean switch — best/early-stop reset, the first epoch after resume sets the new baseline (past
  val-losses were never computed and can't be recovered).
- ✅ Verified on GPU: `--quick` end-to-end (VAL-LOSS computed, best by val-loss) **and a real resume from
  the live ep17 checkpoint** (prints the switch, continues at ep18, saves `best_val_loss`). Backed up the
  SI-SDR-best ep16 weights (+4.88 dB) to `best_sisdr_ep16.pt` before resume overwrites `best.pt`.
- Context: the full run reached **best SI-SDR +4.88 dB @ ep16** (≈ the 40k hq run's +4.89) before the
  earlier commit-limit crash; it can be resumed directly with the new selection in effect.

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
