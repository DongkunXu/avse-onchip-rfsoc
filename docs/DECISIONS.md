# DECISION LOG (ADR-style)

Each significant choice: what was decided, why, by whom, when. Append-only; supersede rather than
delete (mark old ones `SUPERSEDED`). Status: ✅ decided · ◇ open (owner) · 🔬 to-be-decided-by-evidence.

---

### D-1 ✅ Start a new, independent project rather than refactor the reference deployment
**2026-06-25 — owner.** The reference U-Net's 4-bitstream form factor is intrinsic to its
architecture (long time axis + U-Net skips). Escaping it requires re-architecting, not refactoring.
New repo, own git/venv/docs; old project is reference-only.

### D-2 ✅ Time-domain ONLY (for now); STFT/frequency deferred
**2026-06-25 — owner.** Phase 2 stays time-domain. The STFT-mask candidate (C3) is **out of scope**
for now (revisit only if time-domain options underdeliver on the quality-vs-fit Pareto). Rationale:
lower risk, no FPGA (i)STFT, closer to the existing know-how. Frequency domain is the biggest
departure and is parked, not killed.

### D-8 ✅ Phase-2 candidate set + open exploration mandate
**2026-06-25 — owner.** Prototype **C4 (tiled U-Net)** and **C2 (streaming TCN)**; keep **C5
(DDR-staged)** as the Plan-B control. Owner explicitly mandated exploring **additional time-domain
ideas beyond the documented candidates** ("don't be limited by the old docs; think for yourself;
more attempts encouraged"). Added by analysis (scored with the validated model): **C7 Conv-TasNet-style
time-domain mask** (removes the U-Net skip wall — the root cause — while staying time-domain),
plus **C8 recompute-skip** and **C9 compressed-skip** as combinable BRAM-reclaim levers.

### D-3 ✅ Precision locked at int16
**Inherited.** int8/DPU costs −1.6 to −2.9 dB SI-SDR and breaks the quality floor; QAT judged
unlikely to recover enough. See [`reference/dead-ends.md`](reference/dead-ends.md). 16-bit activations
are *why* BRAM dominates — that constraint is accepted, not fought with lower precision.

### D-4 ✅ Single board fixed: RFSoC 4x2 / ZU48DR
**Inherited.** No AI Engines (Versal-only). AIE/Versal offload is a different project, revisited only
if the owner changes hardware.

### D-5 ✅ Binding metric = peak simultaneously-live activation working set (Σ C×T)
**2026-06-25.** This single, analytically-computable quantity is the optimization target and the
spine of the circuits contribution. Phase 1 builds and validates a model for it.

### D-6 ✅ torch from the cu128 wheel index (not the inherited cu121 pins)
**2026-06-25.** Host GPU is RTX 5070 Ti (Blackwell, sm_120); cu121 builds do not support it. The new
venv installs torch/torchvision/torchaudio from `https://download.pytorch.org/whl/cu128`.

### D-7 ✅ Reusable assets migrated, not re-derived; new models written fresh
**2026-06-25.** Architecture-agnostic, high-value code (data pipeline, audio metrics, loss, config
template, teacher-model reference) was migrated to avoid re-doing solid work. **New, hardware-shaped
model architectures are written from scratch** in `src/avse/models/` — the migrated teacher lives in
`src/avse/reference/` for distillation/comparison only, never as the design center.

---

### D-9 ✅ Phase 3 = HLS-fit-first on C7, with placeholder weights
**2026-06-25 — owner.** Take **C7 (Conv-TasNet-style)** to Phase 3. **First confirm the structure
actually fits** via real Vitis HLS C-synth + Vivado synth/P&R reports, **then** come back to iterate /
retrain a high-quality version. Key consequence: fit is structure-driven, not weight-value-driven, so
Phase 3a synthesizes C7 with **placeholder weights** to get the fit answer fast, decoupled from
quality training. (Pareto: C7 +3.79 dB SI-SDR small-run, 0.017 MB working set; C4 is the high-quality
fallback.) Toolchain on this machine is **Vitis HLS / Vivado 2022.2** (not 2024.2) — fine for the fit
check on ZU48DR.

### D-10 ✅ Next: monolithic total-fit synth, then high-quality retrain
**2026-06-25 — owner.** Fit is confirmed (C7 audio: synth 40%/41%, **post-route 46% BRAM / 17% LUT,
200 MHz timing met**). Next, in order: (1) **monolithic integrated synth** — C7 audio + the video
encoder in ONE HLS design → the *real* whole-system single-config fit number (not the audio+video
estimate); (2) **high-quality retrain** of C7 on full data + export real weights. Owner also asked to
**update the docs first**. (Parallelisable: retrain is GPU, monolithic synth is CPU.)

### D-11 ✅ Serialize Vivado P&R and GPU training on this 32 GB host
**2026-06-25.** This machine has **32 GB RAM** (commit limit ~42 GB), not the 196 GB the old project's
notes assumed. Running a Vivado place-and-route concurrently with a GPU training job exhausted the
Windows commit limit: the retrain crashed (error 1455) and Vivado synth hung (2 h at 0 % CPU). **Run
heavy jobs one at a time** — Vivado P&R alone, then training alone. The training harness was also
hardened (subset windows in-place so workers don't copy the full dataset; per-epoch checkpoints).

### D-12 ✅ Skip non-finite-loss batches (silent-target NaN), don't filter data
**2026-06-25.** Some LRS3 windows have an all-zero (silent) target; torch_pesq's PESQ loss returns NaN
on a silent reference, which (via one NaN gradient) poisons the weights. Chosen fix: **skip any batch
whose total loss is non-finite** (no backward/step) and report `nan-skip N` per epoch. Rationale: it is
robust to *any* NaN source, costs negligible data (such windows are rare), and avoids touching the
asteroid/torch_pesq internals or doing an expensive per-window energy scan of the dataset.

### D-13 ✅ Cache the dataset window list on disk
**2026-06-25.** The train-split scan (34.5k scenes → 315k windows) is disk-bound (~13–23 min). The
window list is cached to `.dataset_cache/<split>_<window-params>.json` so reruns/`--resume` start in
~0.2 s. Keyed by split + window params so it self-invalidates on a config change; git-ignored.

### D-14 ✅ Scene-streaming dataset (read each scene once/epoch), not per-window random access
**2026-06-25 — owner directed ("each scene should be read once; fix the core/structure, no patches").**
The map-style `AVSEDataset` opens a window's 3 files per `__getitem__`; with global shuffle over ~315k
windows / ~34.5k scenes (~9 windows/scene), every scene's files were re-opened ~9×/epoch and re-resampled,
pegging the dataset SSD (random small reads, ~40 MB/s, GPU starved 51% of the time). Pre-packing a
sequential copy was ruled out by disk facts (train = 260 GB; D: free = 70 GB). Chosen fix is structural,
not a cache patch: a new `AVSESceneStreamDataset(IterableDataset)` whose unit of work is a **scene** — each
worker reads a disjoint scene shard, opens each scene's files **once/epoch**, reads them whole+sequentially,
resamples once, and slices all its windows from RAM. Shuffling is preserved by a bounded sliding scene pool
(≈128 scenes resident/worker). Per-window decode + normalization are byte-identical to the old path, so
training numerics are unchanged. Verified A/B: GPU util 46%→74% (steady 80–100%), SSD disktime 437%→26%.
Old `AVSEDataset` kept for `verify_data.py`/val; `data_module.py` (Lightning, unused by `train.py`)
left as-is. Rationale for IterableDataset over a sampler+cache bolt-on: a global-shuffle map-style dataset
*structurally* forces re-opens (any per-worker cache is defeated by scattered indices) — the honest fix is
to change the unit of work, per the owner's "no patch-on-patch" rule.

### D-15 ✅ Removed dead migrated-from-reference code; video encoder promoted to a native model component
**2026-06-25 — owner directed ("clean out all old/unused and other-project code; leave nothing useless").**
Phase 2 reached its quality target by training C7 **from scratch** — distillation was never pursued — so
the migrated 0.37 M teacher model was dead weight. Removed `src/avse/reference/{audio_encoder,avse_model,
fusion,__init__}.py`, `src/avse/data/data_module.py` (PyTorch-Lightning DataModule, unused — `train.py`
builds its own loader), and `src/avse/config/reference_base_config.yaml` (old U-Net config). The ONE piece
still load-bearing — `LightweightVideoEncoder` (used by every candidate via `_tcn_common`) — was moved
`avse/reference/video_encoder.py → avse/models/video_encoder.py` and is now a first-class model component,
not "reference." Dropped the now-unused `pytorch-lightning` dependency. This **supersedes D-7's** "teacher
lives in `src/avse/reference/`" (the reusable video pathway was adopted into the new project; the teacher
itself is gone — its facts survive in `docs/reference/`). Safe vs the live training: deleted files were
not in the running process's import graph; verified the cleaned tree imports + builds both models in an
isolated CPU process (C7 308,544 params unchanged). `analysis/` (Phase-1 model) and `docs/reference/`
(inherited facts) are project deliverables, not migrated junk — kept.

### D-16 ✅ Scene-streaming pool bounded by bytes, not scene count
**2026-06-26.** The scene pool in D-14 was capped by scene COUNT (`scene_buffer`), which is the wrong
invariant: LRS3 lengths are long-tailed (a 74 s scene ≈ 25 MB resident — 16 MB video + ~9 MB audio), so
when epoch 1's shuffle clustered long scenes, 128 scenes × 6 workers exhausted the 32 GB host's commit
limit and the full-data run OOM-crashed at the ep0→1 transition (after a healthy ep0, SI-SDR +2.21 dB).
Fixed by bounding the pool with a **per-worker byte budget** (`--pool-mb`, default 160 MB) plus a secondary
count cap (`--scene-buffer`); `fill()` stops at whichever binds and always keeps ≥1 scene, so resident
memory ≤ budget + one scene regardless of scene-size distribution. Both are CLI-tunable. Verified: single
process over the whole shuffled set (incl. longest scenes) peaks at 823 MB flat; a 6-worker run crosses the
ep0→1 boundary with no MemoryError. Per-window numerics unchanged. Refines [D-14].

### D-17 ✅ Model selection / early-stop on val TOTAL LOSS, not SI-SDR alone
**2026-06-26 — owner.** `best.pt` and early-stop tracked **val SI-SDR only**, but SI-SDR is just one term
(weight 0.5) of the training objective alongside PESQ (3.0), STOI (4.0), L1/L2/multiscale/STFT. That
mismatch let the run flirt with early-stop on SI-SDR while PESQ/STOI were still improving, and saved a
`best.pt` that wasn't the all-round best. Fix: select + early-stop on the **validation total loss** (the
exact training objective on held-out data) — the comprehensive criterion the owner asked for. `evaluate()`
now also computes the total loss over the val set (skipping non-finite/silent-target batches, D-12);
SI-SDR/PESQ/STOI are still logged for reporting. **Logic-only change — no retraining**: training dynamics,
loss, optimizer, data are untouched; only which epoch is "best" and when to stop change. Resume-safe: a
pre-val-loss checkpoint (only `best_sisdr`) triggers a clean switch — best/early-stop reset and the first
epoch after resume sets the new baseline (past val-losses were never computed and use different functions
than the eval metrics, so they're unrecoverable). Verified on GPU: `--quick` end-to-end + a real resume
from the live ep17 checkpoint (switches, continues at ep18, saves `best_val_loss`). The SI-SDR-best ep16
weights were backed up to `best_sisdr_ep16.pt` before resume overwrites `best.pt`.

### D-18 ✅ Deployment fixed-point policy: int16 activation/weight datapath; BN/PReLU scalars high-precision
**2026-06-27.** For the deployment-accurate quality measurement (and the matching HLS), the int16 lock
(D-3) is applied where it belongs — the **BRAM-dominating datapath**: activations → `data_t` ap_fixed<16,7>,
MAC weight operands → `wgt_t` ap_fixed<16,5>, I/O PCM → `sample_t` ap_fixed<16,1>, accumulators wide
(`acc_t` ap_fixed<48,22>, treated exact). The **inline per-channel BN/in_norm affine (s,b) and PReLU slopes
are kept high-precision constants**, NOT forced into `wgt_t`. Reason (found, not assumed): exporting the real
weights showed `in_norm`'s fold scale reaches **~102** (low-variance encoder channels) and `bn2` scales reach
~45 — both overflow `wgt_t`'s ±16 range; and these are pointwise constants, not the systolic-array operands
the int16 lock targets (standard practice, cf. TFLite per-channel scales). Measured cost of the policy
(full dev, `tools/eval_deploy.py`): pure int16 quantization −0.330 dB SI-SDR; the HW hardsigmoid mask adds
only −0.085 dB → on-chip **4.984 / 1.632 / 0.742**, still above the FP32 teacher anchor on SI-SDR/STOI. The
emulator (`tools/c7_fixedpoint.py`) and the HLS ROMs both consume the one exported `deploy_weights.npz`, so
they cannot drift; HLS C-sim will cross-check emulator ≡ silicon (Deliverable B). Related: bn1/bn2 are kept
**inline** rather than folded into the dwconv/out_conv because PyTorch zero-pads the bn1 output before the
dwconv — folding to a single per-channel bias would mis-handle the pad boundary (see `hls/DEPLOY_PLAN.md`).

### D-19 ✅ Video encoder synthesized in a conservative ROLLED schedule; throughput optimization deferred
**2026-06-27 — owner-directed framing ("get the full flow + on-board numbers first, no shortcuts on the
computation; performance/throughput optimization comes after").** The first real-weight monolithic csynth
ran ~6 h without finishing. Root-caused to the faithful video encoder: per-loop II=2 pipelining forced wide
(×96) reduction unrolls and bank-conflict analysis on strided shortcut reads into cyclic-partitioned buffers,
and complete-partitioned the `v_fp_w`/`v_tp_w` weight ROMs into registers → scheduling/binding exploded. The
video is **not** the throughput bottleneck (the audio path dominates latency), so it is synthesized **rolled**
(no per-loop pipeline / no array partition). This is **not a shortcut**: the computation is byte-identical (the
HLS C-sim emulator≡silicon check still holds), only the *hardware schedule* is conservative — low resource,
high latency (2.2 s), synthesizes in ~5 min. Audio stays pipelined (II=2, synthesizes in ~3 min, fits 47%
BRAM/14% DSP/28% LUT). **Re-introducing pipelining / unrolling / module reuse on the video for real-time
throughput is a deliberate SEPARATE optimization phase** AFTER the end-to-end flow + on-board measurement is
in hand — it also strengthens the circuits-architecture narrative (the resource×efficiency trade is itself a
contribution). Tracked in the [[hls-synthesis-and-optimization]] memory.

### D-20 🔄 Phase 4 throughput optimization — value-faithful parallelization; gather decoder; BRAM-guarded
**2026-06-28 — owner-authorized (the deferred D-19 phase).** Optimize throughput while finding the
resource×efficiency balance, no shortcut on the computation. Decisions/findings (full log in
`hls/OPTIMIZATION_PLAN.md`):
- **D-19's premise was wrong** (recorded, not hidden): the real csynth report shows the **video encoder is
  86 % of latency**, not the audio path. Rolling the video was the right call to escape the 6 h synth blow-up,
  but its stated reason ("audio dominates") was incorrect — the video was the #1 target.
- **One recipe, applied throughout:** partition the reduction-input buffers on the **channel** dim (cyclic
  factor matched to the wanted II — 7/16), stage each output's **weight row into registers** (LUT/FF, cheap),
  and **unroll the reduction**. Channel→bank, spatial→within-bank address makes even the strided shortcut
  reads conflict-free — avoiding the original 6 h blow-up cause (strided-into-spatial + complete-partitioned
  big ROMs). Frame (`video_in`) and `audio_in` are **cached on-chip** (burst once) to kill the on-board
  un-bursted-DDR penalty (the 11.67 s/window + ~2 % residual).
- **Decoder rewritten scatter→GATHER:** each output computed once from its two analytic contributors
  (`t=s/16+1,k=s%16` and `t=s/16,k=s%16+16`). Bit-identical to the scatter (acc_t sum is order-independent),
  but **structurally hazard-free** — this **eliminates the on-board RMW decoder hazard at its root** (the
  reason the scatter had to be rolled). Supersedes the "roll the scatter" workaround.
- **Spend DSP/LUT, guard BRAM** (the binding resource): all activation partitions use modest cyclic factors
  (not `complete`, which doubles BRAM via bank rounding). Result so far: **2.564 s → ~0.27 s csynth (~9.5×,
  ~4.4× under real-time)**, every step C-sim bit-identical. The **2.5 h csynth is accepted** (front-end
  analysis of the wide unrolls — verified slowness, not a bug: II correct + C-sim PASS), per owner ("慢可接受,
  别为它放弃优化").
- **BLOCKS (TCN core) held** at the BRAM-expensive frontier (II→1 needs complete partition → ~2× BRAM) until
  the integrated **post-route** BRAM is known. Long syntheses run **detached** (`Start-Process`), per the
  [[hls-synthesis-and-optimization]] lesson. Post-route + on-board numbers pending the running build.

## Pending owner gates (forward-looking)

- ~~before Phase 2: D-2~~ → resolved (D-2: time-domain only).
- ~~after Phase 1: pick directions~~ → resolved (D-8: C4 + C2 + C5, + open exploration → C7/C8/C9).
- ~~after Phase 2: pick the operating point~~ → resolved (D-9: **C7**; Pareto confirmed C7 dominant).
- ~~after Phase 3 fit: monolithic total then retrain~~ → resolved (D-10: monolithic P&R done, 80% BRAM
  one config; high-quality retrain in progress).
- ~~full-data run + deployment-accurate quality~~ → resolved (`p2-c7-full` 5.40 FP32; D-18: on-chip int16
  **4.98 dB / 1.63 / 0.742**, validated emulator). **Deliverable A done.**
- **◇ OPEN (Deliverable B)**: make the HLS value-faithful (DEPLOY_PLAN G1–G11) + load the real weight ROMs,
  C-sim to confirm **emulator ≡ HLS**, then re-run csynth + P&R for the final real-weight fit numbers.
