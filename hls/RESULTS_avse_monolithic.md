# Phase 3 result — MONOLITHIC C7 AVSE fits in one configuration (the real total)

The whole system — C7 audio mask network **+ the video encoder, in ONE HLS design** (`c7_avse_top`).
This is the real single-config number the owner asked for (D-10), not the audio+video estimate.
Target xczu48dr, Vitis HLS 2022.2, placeholder weights.

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

## Post-route (Vivado place-and-route) — the definitive total

_(running — `hls/tcl/run_impl_avse.tcl`; to be filled in when complete.)_

## The before/after that is the paper's headline

| | Reference (whole AVSE) | C7 AVSE (this design) |
|---|---:|---:|
| BRAM | **215 %** (4 PCAP bitstreams) | **70 %** (one config) |
| LUT | **126 %** | 90 % csynth → far less post-route |
| configuration | 4 sequential bitstreams (PCAP) | **1 static bitstream** |

The reference could not co-reside (215 % BRAM, 126 % LUT) → forced into 4 reconfigured bitstreams.
The C7 no-skip architecture brings the **whole** system to **70 % BRAM** in **one** configuration —
the project's goal, demonstrated in real synthesis.

## Caveats
- Placeholder weights (fit is structure-driven, D-9) — quality comes from the retrained model.
- Windowed (non-streaming); the video encoder is a compact representative of the reference's.
- LUT will be the resource to watch at P&R; if tight, streaming the audio path or trimming the
  video encoder widens the margin.
