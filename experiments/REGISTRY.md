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
on-board (rolled). A small (−0.22 dB, corr 0.9855, quality-negligible) silicon-vs-design residual remains.
Post-route: BRAM 85% / LUT 19% / DSP 17% / 200 MHz. **The project's central goal is demonstrated on real
silicon enhancing speech.**

### Phase 4 throughput optimization (2026-06-28) — same single static bitstream, **40.8× faster on-board**

The optimized design (HLS parallelization: on-chip frame/audio caches, unrolled conv/TCN reductions with
channel-partition + register weights, **gather decoder** replacing the rolled scatter — all C-sim
bit-identical) rebuilt end-to-end (csynth → P&R → bitstream → board) and run on the **same 16 windows**:

| | SI-SDR | PESQ-WB | STOI | note |
|---|---:|---:|---:|---|
| **on-board OPTIMIZED** | **+6.66** | **1.72** | **0.72** | **286 ms/window (40.8× vs the 11.67 s baseline)** |
| on-board baseline (rolled) | +6.66 | 1.72 | 0.72 | 11.67 s/window — identical quality |
| emulator (same 16 win) | +6.88 | 1.73 | 0.72 | silicon matches to **corr 0.9855** (= baseline) |

**Post-route (optimized): BRAM 76.8 % (↓ from 85.3 %), DSP 36.6 %, LUT 32.1 %, FF 10.4 %, 200 MHz met
(WNS +0.083 ns).** Latency 0.269 s csynth (9.5×) / **0.286 s on-board (40.8×) → 4.2× under real-time.** The
optimization spent DSP/LUT/FF and *freed* BRAM (the binding resource); quality unchanged on silicon. The
on-board caches removed the un-bursted-DDR penalty (so silicon tracks csynth, not 4.5× over). The ~17%-rel
silicon-vs-emulator residual (corr 0.9855, quality-negligible) is **unchanged from baseline** — pre-existing,
NOT the DDR reads (a silicon-vs-C-sim effect; co-sim item). The decoder hazard is now eliminated at the root
(gather, no scatter). **The whole AVSE runs faster than real-time on one static bitstream, vs the reference's
4 PCAP bitstreams.**

### Stratified per-SNR-bin on-board eval (2026-06-28) — the representative, comparable number

Mirrors the reference 10-bin SNR protocol (`test reference/selection_manifest.json`: 10 bins × 2.5 dB,
−15…+10 dB, seed 42), **20% of scenes sampled per bin** (665 scenes / 4917 windows), run on the optimized
FPGA, scored per scene → per bin → **scene-count-weighted** (tools: `prep_board_snr_bins.py`,
`run_board_chunks.sh`, `score_board_snr_bins.py`, `plot_snr_bins.py`). Our validated windowing/normalization.

| input SNR (dB) | n | FPGA SI-SDR | PESQ | STOI | mixed SI-SDR | Δ SI-SDR |
|---|--:|--:|--:|--:|--:|--:|
| [−15,−12.5] | 36 | −5.72 | 1.150 | 0.531 | −11.99 | +6.27 |
| [−12.5,−10] | 38 | −4.74 | 1.139 | 0.553 | −10.56 | +5.82 |
| [−10,−7.5] | 82 | 0.51 | 1.337 | 0.629 | −4.99 | +5.50 |
| [−7.5,−5] | 89 | 1.72 | 1.449 | 0.687 | −2.25 | +3.97 |
| [−5,−2.5] | 81 | 4.27 | 1.497 | 0.746 | 0.05 | +4.22 |
| [−2.5,0] | 85 | 6.10 | 1.665 | 0.788 | 4.57 | +1.53 |
| [0,2.5] | 87 | 6.34 | 1.705 | 0.776 | 5.80 | +0.54 |
| [2.5,5] | 86 | 7.91 | 1.814 | 0.800 | 7.86 | +0.05 |
| [5,7.5] | 45 | 13.31 | 2.223 | 0.876 | 13.41 | −0.10 |
| [7.5,10] | 36 | 15.30 | 2.322 | 0.904 | 15.59 | −0.29 |
| **weighted (by bin scenes)** | **665** | **4.59** | **1.615** | **0.735** | **1.95** | **+2.64** |

Clean monotonic SNR trend; the model does its work at low/mid SNR (+5–6 dB SI-SDR in the −15…−7.5 bins) and
≈preserves above ~+5 dB. **FPGA is value-faithful per bin**: the int16 **software (emulator)** on the same
windows = weighted **4.80 / 1.618 / 0.738**, so FPGA trails by **−0.21 dB** (= the established silicon-vs-
emulator gap, corr 0.9855). The weighted overall reconciles with the full-dev emulator (4.98) within the 20%
sampling; proportional sampling → weighted ≡ simple mean (both 4.59). Plot: `hw/board/snr_eval/
snr_trend_onboard.png`; full JSON: `snr_bin_results.json`. (FP32 per-bin upper bound is a quick GPU add-on.)

## Reference anchors (for comparison, not experiments)

| name | SI-SDR | PESQ-WB | STOI | note |
|---|---|---|---|---|
| Reference FP32 AV (PAPER_DATA §B, N=3319) | +3.99 | 1.673 | 0.741 | the teacher's quality; C7 already ≈ this after a small run |
| Deployed INT16 AV (PAPER_DATA §B, N=496) | +5.46 | 1.743 | 0.738 | reference FPGA; working set 4.1 MB, does NOT fit single-config |

> Caveat: anchor metrics are from PAPER_DATA on different eval subsets than the p2-* runs (dev-160).
> The comparison is indicative for screening, not a controlled head-to-head. C7/C2 are small runs
> (10 epochs / 10k windows) and still improving — their quality is a lower bound.
