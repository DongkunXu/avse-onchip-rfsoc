# Vitis HLS 2022.2 C-synthesis reports — the C7 AVSE design

Tool-generated `csynth.rpt` files, copied verbatim from the git-ignored `hls/build/*/sol1/syn/report/`.
Target `xczu48dr-ffvg1517-2-e`, **HLS target clock 5 ns (200 MHz)** — note this is the *synthesis target*;
the implemented bitstream clock is **187.5 MHz** (see `hw/reports/`, the optimized design's critical path
is 5.250 ns and does not close 200 MHz). HLS resource numbers are **estimates** and are pessimistic for
LUT/FF; the authoritative post-implementation resources are in `hw/reports/`.

| file | solution | what it is |
|---|---|---|
| `c7_avse_opt_csynth.rpt` | `c7_avse_opt` | **the Phase-4 optimized monolith** (audio+video, one design) — the deployed design |
| `c7_avse_baseline_csynth.rpt` | `c7_avse` | the rolled baseline monolith (pre-optimization) — the 9.5× reference |
| `c7_fit_placeholder_csynth.rpt` | `c7_fit` | Phase-3a structural fit (placeholder weights) — proved one-config fit |
| `c7_video_component_csynth.rpt` | `c7_video` | video-encoder component (substantiates "video was 86 % of baseline latency") |
| `c7_audio_opt_component_csynth.rpt` | `c7_audio_opt` | optimized audio-core component (standalone tuning synth) |

## Headline numbers (top function `c7_avse_top`, read straight from the reports)

| | **optimized** (`c7_avse_opt`) | baseline (`c7_avse`) | ratio |
|---|---:|---:|---:|
| Latency (cycles) | **53,742,170** | 512,848,453 | **9.54× fewer** |
| Est. time @ 5 ns target | 0.269 s | 2.564 s | — |
| **Real time @ 187.5 MHz (board)** | **0.287 s ≈ 286 ms** | — | — |
| BRAM_18K (HLS est.) | 1687 (78 %) | 1561 (72 %) | |
| DSP (HLS est.) | 1565 (36 %) | 833 (19 %) | |
| LUT (HLS est., pessimistic) | 330,374 (77 %) | 188,035 (44 %) | |

> The optimization trades cycles for parallelism: **9.5× fewer cycles** at a modestly longer critical path
> (clock 200→187.5 MHz), netting **40.8× faster on-board** (board-vs-board, 11.67 s → 0.286 s). Every step is
> C-sim bit-identical to the baseline (value-faithful) — see `hls/OPTIMIZATION_PLAN.md`. Note the HLS
> BRAM_18K count (1687) differs in unit from the Vivado BRAM-tile count (829 × 36 Kb tiles); the binding
> post-route resource is **76.8 % BRAM** in `hw/reports/`.

Regenerate any report: the `hls/tcl/run_csynth_*.tcl` scripts (need `hls/src/c7_weights.hpp`, tracked).
