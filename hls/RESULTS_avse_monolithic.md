# Phase 3 result — MONOLITHIC C7 AVSE fits in one configuration (the real total)

The whole system — C7 audio mask network **+ the video encoder, in ONE HLS design** (`c7_avse_top`).
This is the real single-config number the owner asked for (D-10), not the audio+video estimate.
Target xczu48dr, Vitis HLS 2022.2.

---

## ✅ REAL-WEIGHT DEPLOYED DESIGN (definitive — Phase 3b)

The trained `p2-c7-full/best.pt` weights, BN-folded into `hls/src/c7_weights.hpp`, with the value-faithful
HLS (audio core inline bn1/bn2 + corrected decoder offset; faithful video encoder). The computation is
**HLS-C-sim-validated bit-faithful to the fixed-point emulator** that measured the on-chip quality
(**SI-SDR 4.984 / PESQ 1.632 / STOI 0.742**, full dev). Video encoder + VPROJ/VUP synthesized in a
conservative **rolled** schedule (D-19; throughput optimization is a deliberate later phase).

| Resource | csynth | **post-route** | Available | **Util** | (placeholder post-route) |
|---|---:|---:|---:|---:|---:|
| **BRAM_18K** | 1603 | **1843** | 2160 | **85.3 %** | 80.3 % |
| **LUT** | 189,313 | **80,933** | 425,280 | **19.0 %** | 41.4 % |
| **DSP** | 833 | **720** | 4272 | **16.9 %** | 20.2 % |
| FF | 53,192 | 41,711 | 850,560 | 4.9 % | 11.8 % |
| URAM | 0 | 0 | 80 | 0 % | 0 % |

**Timing MET** — post-route CP 4.869 ns < 5.000 ns → **200 MHz**, WNS +0.131 ns. Packaged IP: `export.zip`.
Latency 2.53 s/window (rolled video dominates — the throughput-optimization target, D-19).

> **The complete AVSE with the REAL trained weights, computation C-sim-validated, fits ONE static FPGA
> configuration and closes timing: 85 % BRAM, 19 % LUT, 17 % DSP, 200 MHz.** BRAM (the activation/weight
> wall) is the binding resource at 85 % — ~15 % headroom; real weight ROMs raise it vs the placeholder
> (80 %), while the rolled video drops LUT (41 % → 19 %). This is the project's central goal, achieved with
> the actual deployed model on real place-and-route.

---

## Historical: structure-fit proof with placeholder weights (D-9)

Below is the original fit proof (placeholder weights, pipelined video) that established the structure fits
before the real weights existed. Kept for the record.

## Whole-system csynth estimate (`c7_avse_top_csynth.rpt`)

| Resource | Used | Available | Util |
|---|---:|---:|---:|
| **BRAM_18K** | 1505 | 2160 | **69.7 %** |
| **LUT** | 382,435 | 425,280 | **89.9 %** |
| DSP | 1200 | 4272 | 28.1 % |
| FF | 201,955 | 850,560 | 23.7 % |
| URAM | 0 | 80 | 0 % |

**The complete AVSE fits in one static configuration** (every resource < 100 %). BRAM 70 % is
comfortable; LUT 90 % is the csynth *estimate* — and HLS over-counts LUT badly pre-route (the
standalone C7 audio went 41 % csynth → **17 %** post-route, a 2.4× drop). Post-route LUT is expected
far lower. **Post-route numbers below (Vivado P&R) are the definitive total.**

## Post-route (Vivado place-and-route) — the definitive total ✅

`hls/tcl/run_full_avse.tcl` (fresh csynth + Vivado 2022.2 P&R, run alone), exit 0.

| Resource | Post-route | Available | Util | (csynth est.) |
|---|---:|---:|---:|---:|
| **BRAM** | 1735 | 2160 | **80.3 %** | 69.7 % |
| **LUT** | 175,942 | 425,280 | **41.4 %** | 89.9 % |
| DSP | 864 | 4272 | 20.2 % | 28.1 % |
| FF | 100,670 | 850,560 | 11.8 % | 23.7 % |
| URAM | 0 | 80 | 0 % | 0 % |

**Timing MET** — post-route 4.870 ns < 5.000 ns → **the whole AVSE closes at 200 MHz**.

> **The complete audio-visual speech enhancement system — C7 audio mask network + video encoder, in
> ONE static FPGA configuration — fits and closes timing: 80 % BRAM, 41 % LUT, 20 % DSP, 200 MHz.**
> LUT collapsed from the 90 % csynth estimate to 41 % post-route (HLS over-counts pre-route). BRAM
> (80 %) is the binding resource, with ~20 % headroom. The project's central goal is achieved on real
> place-and-route reports.

Note: placeholder weights occupy the same-sized ROMs as trained weights, so these resource numbers
are representative of the real model (weights affect output quality, not footprint).

## The before/after that is the paper's headline

| | Reference (whole AVSE) | C7 AVSE (this design, post-route) |
|---|---:|---:|
| BRAM | **215 %** (impossible → 4 PCAP bitstreams) | **80 %** (one config) |
| LUT | **126 %** | **41 %** |
| DSP | — | 20 % |
| timing | — | **200 MHz, met** |
| configuration | 4 sequential bitstreams (PCAP reconfig) | **1 static bitstream** |

The reference could not co-reside (215 % BRAM, 126 % LUT) → forced into 4 reconfigured bitstreams.
The C7 no-skip architecture brings the **whole** system to **70 % BRAM** in **one** configuration —
the project's goal, demonstrated in real synthesis.

## Caveats
- Placeholder weights (fit is structure-driven, D-9) — quality comes from the retrained model.
- Windowed (non-streaming); the video encoder is a compact representative of the reference's.
- LUT will be the resource to watch at P&R; if tight, streaming the audio path or trimming the
  video encoder widens the margin.
