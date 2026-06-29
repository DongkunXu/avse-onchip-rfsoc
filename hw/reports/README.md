# Vivado 2022.2 post-implementation reports — the single static AVSE bitstream (optimized, Phase 4)

These are the **tool-generated** reports for `avse_sys.bit` (target `xczu48dr-ffvg1517-2-e`, RFSoC 4x2),
copied verbatim from `hw/vivado_proj/avse_sys.runs/impl_1/` (which is git-ignored as a build dir). They
are the **authoritative** resource/timing numbers — the markdown summaries elsewhere transcribe from here.

| file | what it is |
|---|---|
| `postimpl_utilization.rpt` | placed-stage resource utilization (cell counts are final post-route) |
| `postroute_timing_summary.rpt` | routed timing (clocks, WNS/WHS) |
| `postroute_power.rpt` | routed power estimate |
| `route_status.rpt` | routing completion (0 unrouted nets) |

## The real numbers (read them straight from the reports)

**Resources** (`postimpl_utilization.rpt`):

| | used | available | util |
|---|---:|---:|---:|
| Block RAM tile | **829** | 1080 | **76.76 %** |
| DSP48E2 | **1563** | 4272 | **36.59 %** |
| CLB LUT | **136,675** | 425,280 | **32.14 %** |
| FF (register) | **88,734** | 850,560 | **10.43 %** |
| URAM | 0 | 80 | 0 % |

**Timing** (`postroute_timing_summary.rpt`): one clock `clk_pl_0`, **period 5.333 ns = 187.5 MHz**,
**WNS +0.083 ns**, 0 failing endpoints — *"All user specified timing constraints are met."*

> ⚠️ **Clock = 187.5 MHz, not 200 MHz.** The throughput optimization lengthened the critical path to
> 5.250 ns (= 5.333 − 0.083), so the optimized design does **not** close 200 MHz (5.0 ns); the PL clock
> `clk_pl_0` is 187.5 MHz and timing is met there. This is consistent end-to-end: the board measured
> **286.2 ms/window**, and 53,742,170 cycles (the HLS cycle count) × 5.333 ns = **286.6 ms** — an exact
> match. (The HLS csynth *estimate* of 0.269 s assumed its 5 ns / 200 MHz target clock; the implemented
> clock is 187.5 MHz, so the real per-window time is 0.287 s = the board.) The speedup factors are
> unaffected: **9.5×** is the cycle-count ratio vs the rolled baseline, **40.8×** is board-vs-board
> (11.67 s → 0.286 s), **4.2×** under the 1.2 s real-time budget.

Regenerate: `hw/rebuild_vivado_opt.bat` (Vivado 2022.2) → reports land in `avse_sys.runs/impl_1/`.
The 34 MB `avse_sys.bit` itself is not tracked (regenerable); see `hw/README.md`.
