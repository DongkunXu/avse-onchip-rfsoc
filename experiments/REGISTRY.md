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

**Post-route (optimized): BRAM 76.8 % (↓ from 85.3 %), DSP 36.6 %, LUT 32.1 %, FF 10.4 %; timing met at
187.5 MHz (`clk_pl_0` 5.333 ns, WNS +0.083 ns).** Tool reports tracked in `hls/reports/` + `hw/reports/`.
Cycle count 9.5× below baseline (512.8M→53.7M cyc); **on-board 0.286 s/window (40.8×) → 4.2× under real-time**
— cycle-exact (53.7M × 5.333 ns = 286 ms). The optimized critical path (5.250 ns) does not close 200 MHz; the
0.269 s csynth figure was the HLS estimate at its 5 ns target. The optimization spent DSP/LUT/FF and *freed*
BRAM (the binding resource); quality unchanged on silicon. The ~17%-rel
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
≈preserves above ~+5 dB.

**Three realizations on the identical 665-scene set (scene-count-weighted overall):**

| realization | SI-SDR | PESQ-WB | STOI | vs prev |
|---|--:|--:|--:|---|
| **FP32** (original `best.pt`, float inputs) | **5.22** | 1.712 | 0.750 | — (≈ full-dev 5.40, within 20% sampling) |
| **int16 quant. emulation** (same int16 inputs as the chip) | **4.80** | 1.618 | 0.738 | −0.42 dB (quantization cost) |
| **on-board FPGA** (the optimized bitstream) | **4.59** | 1.615 | 0.735 | −0.21 dB (= the known silicon-vs-emu gap, corr 0.9855) |

So FP32 → silicon is **−0.63 dB SI-SDR** total, monotonic and consistent at every bin (the FPGA and emulator
curves nearly overlap → value-faithful across the whole SNR range). 3-panel plot (FP32 / int16-quant dashed /
on-board, with weighted-average lines): `hw/board/snr_eval/snr_trend_onboard.png`. Full JSON +per-bin for all
three: `snr_bin_results.json`. Tools: `eval_fp32_snr_bins.py`, `plot_snr_bins.py`. **Note:** FP32 uses
full-precision inputs (true upper bound); FPGA/emulator use the int16 inputs the chip consumes.

### Visual ablation — does the model actually use the video? (2026-06-29) — FP32 *and* on-chip FPGA

To confirm the **visual modality contributes** (not an audio-only network in disguise), an A/B on the
**identical 665-scene SNR-bin set**, done in **both realms**: for every window, run the model twice with the
**same audio** — once with the real video, once with the video **zeroed** (a "black screen": time-varying lip
motion removed, only the learned static prior kept — the conservative ablation). Same windowing /
normalization / metrics / per-scene→bin→scene-count-weighted aggregation as everywhere else. Both with-video
arms **reproduce their stored rows to 3 decimals** (FP32 5.216/1.712/0.750; FPGA 4.592/1.615/0.735) —
built-in sanity checks.
- **FP32 (Python):** `torch.zeros_like` video. Tools `eval_video_ablation_snr_bins.py`, `plot_video_ablation.py --realm fp32`.
- **On-chip (FPGA):** the **same optimized bitstream**, video DDR buffer zeroed on-board (int16 0 == float 0 ==
  the same black screen), same audio re-fed. Driver `run_fpga.py --zero-video`; tools
  `prep_board_novideo_chunks.py`, `run_board_chunks.sh … --zero-video`, `score_board_novideo.py`,
  `plot_video_ablation.py --realm fpga`. 4917 windows @ 286 ms = ~27.5 min board time. Outputs differ 25–47 %
  from with-video per window (the flag genuinely changes the compute).

**FP32 (Python) — with video → video zeroed:**

| input SNR (dB) | n | SI-SDR | Δ | PESQ | Δ | STOI | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| [−15,−12.5] | 36 | −5.62→−8.30 | −2.68 | 1.163→1.091 | −0.072 | 0.544→0.478 | −0.066 |
| [−12.5,−10] | 38 | −4.64→−6.94 | −2.30 | 1.156→1.092 | −0.064 | 0.567→0.518 | −0.049 |
| [−10,−7.5] | 82 | 1.12→−1.51 | −2.63 | 1.397→1.284 | −0.113 | 0.652→0.587 | −0.065 |
| [−7.5,−5] | 89 | 2.16→−1.12 | −3.29 | 1.512→1.327 | −0.185 | 0.702→0.613 | −0.089 |
| [−5,−2.5] | 81 | 4.70→1.55 | −3.15 | 1.570→1.413 | −0.157 | 0.762→0.675 | −0.086 |
| [−2.5,0] | 85 | 6.88→2.81 | −4.07 | 1.771→1.506 | −0.265 | 0.803→0.696 | −0.108 |
| [0,2.5] | 87 | 6.95→2.61 | −4.34 | 1.813→1.512 | −0.301 | 0.790→0.686 | −0.104 |
| [2.5,5] | 86 | 8.83→3.33 | −5.49 | 1.948→1.526 | −0.422 | 0.815→0.682 | −0.133 |
| [5,7.5] | 45 | 14.25→9.67 | −4.58 | 2.418→2.045 | −0.373 | 0.890→0.829 | −0.061 |
| [7.5,10] | 36 | 16.48→9.20 | −7.28 | 2.548→1.987 | −0.561 | 0.921→0.828 | −0.093 |
| **weighted** | **665** | **5.22→1.29** | **−3.93** | **1.712→1.463** | **−0.249** | **0.750→0.660** | **−0.090** |

**On-chip FPGA (same bitstream) — with video → video zeroed:**

| input SNR (dB) | n | SI-SDR | Δ | PESQ | Δ | STOI | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| [−15,−12.5] | 36 | −5.72→−8.16 | −2.44 | 1.150→1.091 | −0.059 | 0.531→0.475 | −0.056 |
| [−12.5,−10] | 38 | −4.74→−6.73 | −1.99 | 1.139→1.098 | −0.041 | 0.553→0.510 | −0.042 |
| [−10,−7.5] | 82 | 0.51→−2.07 | −2.57 | 1.337→1.229 | −0.109 | 0.629→0.565 | −0.064 |
| [−7.5,−5] | 89 | 1.72→−1.72 | −3.44 | 1.449→1.271 | −0.178 | 0.687→0.590 | −0.097 |
| [−5,−2.5] | 81 | 4.27→0.94 | −3.33 | 1.497→1.341 | −0.156 | 0.746→0.654 | −0.093 |
| [−2.5,0] | 85 | 6.10→1.99 | −4.11 | 1.665→1.407 | −0.258 | 0.788→0.676 | −0.112 |
| [0,2.5] | 87 | 6.34→1.76 | −4.59 | 1.705→1.417 | −0.288 | 0.776→0.661 | −0.115 |
| [2.5,5] | 86 | 7.91→2.26 | −5.65 | 1.814→1.421 | −0.394 | 0.800→0.653 | −0.147 |
| [5,7.5] | 45 | 13.31→8.32 | −4.99 | 2.223→1.827 | −0.396 | 0.876→0.793 | −0.084 |
| [7.5,10] | 36 | 15.30→7.22 | −8.08 | 2.322→1.714 | −0.608 | 0.904→0.786 | −0.118 |
| **weighted** | **665** | **4.59→0.53** | **−4.06** | **1.615→1.372** | **−0.243** | **0.735→0.637** | **−0.098** |

**Visual contribution (weighted Δ, removing video):**

| realm | ΔSI-SDR | ΔPESQ | ΔSTOI |
|---|--:|--:|--:|
| FP32 (Python) | **−3.93 dB** | **−0.249** | **−0.090** |
| on-chip FPGA  | **−4.06 dB** | **−0.243** | **−0.098** |

**The visual contribution is large, consistent on all three metrics, present in every single bin (no
cherry-picking), and — crucially — fully preserved on real silicon** (FPGA Δ ≈ FP32 Δ to within ~0.1 dB /
0.006 PESQ / 0.008 STOI). The gap *widens with SNR* (−2…−2.7 dB in the lowest bins → −4.5…−8 dB above
+2.5 dB): when the audio is less catastrophically corrupted the lip stream is more fully exploitable, whereas
at very low SNR even AV is hard. **The model is genuinely audio-visual, and that benefit survives quantization
+ the silicon datapath.** Plots: `hw/board/snr_eval/{video_ablation,board_video_ablation,video_ablation_combined}.png`.
Full per-bin JSON: `video_ablation_results.json` (FP32), `board_video_ablation_results.json` (FPGA).

## Reference anchors (for comparison, not experiments)

| name | SI-SDR | PESQ-WB | STOI | note |
|---|---|---|---|---|
| Reference FP32 AV (PAPER_DATA §B, N=3319) | +3.99 | 1.673 | 0.741 | the teacher's quality; C7 already ≈ this after a small run |
| Deployed INT16 AV (PAPER_DATA §B, N=496) | +5.46 | 1.743 | 0.738 | reference FPGA; working set 4.1 MB, does NOT fit single-config |

> Caveat: anchor metrics are from PAPER_DATA on different eval subsets than the p2-* runs (dev-160).
> The comparison is indicative for screening, not a controlled head-to-head. C7/C2 are small runs
> (10 epochs / 10k windows) and still improving — their quality is a lower bound.
