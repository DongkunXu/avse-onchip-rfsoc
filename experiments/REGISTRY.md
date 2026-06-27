# EXPERIMENT REGISTRY

One row per training run (Phase 2). Keep the **table** here as the index; put per-run detail (config
snapshot, full metrics, notes, checkpoint pointer) under `experiments/<exp_id>/`. Checkpoints
themselves are git-ignored — store a path/pointer, not the blob.

## How to log an experiment

1. Pick an `exp_id`: `p2-<NNN>-<short-slug>` (e.g. `p2-001-streaming-tcn-baseline`).
2. Create `experiments/<exp_id>/` with: `config.yaml` (snapshot), `metrics.json` (final numbers),
   `notes.md` (what/why/observations), and a `checkpoint.txt` pointing to the `.ckpt` location.
3. Add a row below. Tie it to the analysis candidate it implements and the working-set estimate it
   was supposed to hit (so software reality can be checked against the Phase-1 prediction).

## Registry

| exp_id | date | candidate / arch | deployable working set | SI-SDR | PESQ-WB | STOI | params | status | notes |
|---|---|---|---|---|---|---|---|---|---|
| **p2-c7-full** | 2026-06-27 | C7 Conv-TasNet (mask) | 0.017 MB | **+5.40** | **1.727** | **0.754** | 308,544 | **done** | full data (315k win), early-stop ep23 on val total-loss, best=ep18. **Metrics are full-dev (3327 scenes, `tools/eval_full_dev.py`)** — beats the FP32 teacher anchor (3.99/1.673/0.741, N=3319) on all three at 1/240 the working set, single-config. Next: export best.pt → HLS ROMs. |
| p2-c7-hq | 2026-06-25 | C7 Conv-TasNet (mask) | 0.017 MB | **+4.89** (best, ep16) | **1.683** | **0.718** | 308,544 | done | 20ep/40k win, cosine LR; final ep19 +4.79; **> reference FP32 (+3.99)** at 1/240 the working set. Next: full-data run. |
| p2-c7-r1 | 2026-06-25 | C7 Conv-TasNet (mask) | 0.017 MB | +3.79 | 1.565 | 0.690 | 308,544 | done | 10ep/10k win (first run) |
| p2-c2-r1 | 2026-06-25 | C2 streaming-TCN (mapping) | 0.033 MB | +1.12 | 1.478 | 0.672 | 343,616 | done | 10ep/10k win; dominated by C7 → masking > direct mapping |

## Deployment-accurate (int16 fixed-point) — `tools/eval_deploy.py`

Same full-dev protocol as the table above, but the forward is the bit-faithful fixed-point emulator
(`tools/c7_fixedpoint.py`) built from `deploy_weights.npz` (exported from `p2-c7-full/best.pt`). Validated:
the emulator's fp+sigmoid path reproduces the FP32 row (5.399/1.727/0.754) to 3 decimals.

| config | SI-SDR | PESQ-WB | STOI | note |
|---|---:|---:|---:|---|
| FP32 (sigmoid) | +5.399 | 1.727 | 0.754 | = the p2-c7-full row (chain check) |
| int16 (sigmoid) | +5.069 | 1.634 | 0.746 | pure quantization cost −0.330 dB |
| **int16 + hardsigmoid (on-chip)** | **+4.984** | **1.632** | **0.742** | the deployed number; hardsigmoid adds −0.085 dB |

On-chip C7 beats the FP32 teacher anchor (SI-SDR +0.99, STOI +0.001) at 1/240 the working set, single static
config (83% BRAM post-route). HLS C-sim confirmed emulator ≡ HLS-C++ to 0.85% (B2).

## On real hardware (RFSoC 4x2, single static bitstream) — `hw/board/run_fpga.py`

The whole real-weight AVSE built end-to-end (HLS → P&R → bitstream → board) and run on the FPGA. 16-window /
2-scene subset (board compute 11.67 s/window — un-optimized rolled video):

| | SI-SDR | PESQ-WB | STOI | note |
|---|---:|---:|---:|---|
| **on-board (FPGA)** | **+6.66** | **1.72** | **0.72** | real silicon, one static bitstream |
| emulator (same 16 win) | +6.88 | 1.73 | 0.72 | board matches to **−0.22 dB**, corr 0.9855 |
| mixed input (baseline) | +4.40 | 1.50 | 0.62 | board beats it by **+2.27 dB** |

A decoder scatter-accumulate pipeline hazard (periodic corruption, C-sim-invisible) was found and fixed
on-board. A small (~2% rms, −0.22 dB, quality-negligible) silicon-vs-design residual remains — deferred to
the throughput-optimization phase (un-bursted video DDR reads). Post-route: BRAM 83% / LUT 20% / DSP 17% /
200 MHz. **The project's central goal is demonstrated on real silicon enhancing speech.**

## Reference anchors (for comparison, not experiments)

| name | SI-SDR | PESQ-WB | STOI | note |
|---|---|---|---|---|
| Reference FP32 AV (PAPER_DATA §B, N=3319) | +3.99 | 1.673 | 0.741 | the teacher's quality; C7 already ≈ this after a small run |
| Deployed INT16 AV (PAPER_DATA §B, N=496) | +5.46 | 1.743 | 0.738 | reference FPGA; working set 4.1 MB, does NOT fit single-config |

> Caveat: anchor metrics are from PAPER_DATA on different eval subsets than the p2-* runs (dev-160).
> The comparison is indicative for screening, not a controlled head-to-head. C7/C2 are small runs
> (10 epochs / 10k windows) and still improving — their quality is a lower bound.
