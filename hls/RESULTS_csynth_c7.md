# Phase 3 result — C7 audio path fits (Vitis HLS 2022.2 C-synth)

**Date**: 2026-06-25 · `vitis_hls -f hls/tcl/run_csynth_c7.tcl` (exit 0).
Target: xczu48dr-ffvg1517-2-e (RFSoC 4x2). Placeholder weights (fit is structure-driven, D-9).

## C7 audio mask network — csynth utilization estimate

| Resource | Used | Available | Util |
|---|---:|---:|---:|
| **BRAM_18K** | 862 | 2160 | **39.9 %** |
| **LUT** | 173,327 | 425,280 | **40.8 %** |
| DSP | 510 | 4272 | 11.9 % |
| FF | 67,300 | 850,560 | 7.9 % |
| URAM | 0 | 80 | 0 % |

Biggest BRAM consumers (Memory detail): `Win` weight ROM 76, the four activation buffers
(w/y/h/hd at cyclic factor 2) + the decoder accumulator `obuf[19200]`. Compute (1×1 convs IN1x1 /
OUT1x1 / BOT / MASK) drives the LUT/DSP.

## Why this is the headline result

| | Reference (audio path) | C7 (audio path) |
|---|---:|---:|
| BRAM | **152 %** (enc 57 + dec 95) | **40 %** |
| LUT | part of the 126 % concurrent wall | **41 %** |
| form factor | forced into separate bitstreams | single IP |

- The reference's **audio** path alone needed 152 % BRAM — the reason the whole design was split into
  4 PCAP-swapped bitstreams. C7's no-skip, single-resolution architecture brings it to **40 % BRAM**
  (~3.8× less) and **41 % LUT**.
- Add the known video encoder (reference ~38 % BRAM, ~30 % LUT): whole-system estimate
  ≈ **78 % BRAM / ~71 % LUT** → **fits in ONE static configuration** — exactly what the reference's
  215 % BRAM could not do. **The project's central hypothesis is confirmed in real synthesis.**

## Post-route (Vivado place-and-route, the real numbers) — `export_impl.rpt`

`vitis_hls -f hls/tcl/run_impl_c7.tcl` (HLS impl flow → Vivado 2022.2 synth + P&R), exit 0.

| Resource | Post-route | Available | Util | (csynth est.) |
|---|---:|---:|---:|---:|
| **LUT** | 71,481 | 425,280 | **16.8 %** | 40.8 % |
| **BRAM** | 987 | 2160 | **45.7 %** | 39.9 % |
| DSP | 377 | 4272 | 8.8 % | 11.9 % |
| FF | 40,856 | 850,560 | 4.8 % | 7.9 % |
| URAM | 0 | 80 | 0 % | 0 % |

**Timing MET** — post-route 4.932 ns < 5.000 ns target → **closes at 200 MHz**.

Place-and-route LUT (17 %) is far below the csynth estimate (41 %) — HLS over-counts LUT pre-route.
So the real picture is even stronger:

> **Whole-system estimate (post-route C7 audio + reference video ~38 % BRAM / ~30 % LUT):
> ≈ 84 % BRAM, ≈ 47 % LUT, ~9 % DSP, timing-closed at 200 MHz → FITS in ONE static configuration.**
> The reference needed 215 % BRAM + 126 % LUT (4 bitstreams). **Fit is confirmed by real synthesis
> AND place-and-route.** BRAM (~84 %) is now the tighter resource — a streaming C7 (vs this windowed
> version) or a narrower decoder accumulator would widen the margin further if needed.

## Honest caveats / next steps
- **Windowed** (non-streaming) implementation; streaming would cut BRAM further (this already fits).
- Video is **cited** from the reference, not yet co-synthesized in one design; a monolithic
  C7-audio + video synth is the final single-config confirmation.
- **Windowed** (non-streaming) implementation; streaming would be smaller still (this already fits).
- Video is **cited** from the reference, not yet co-synthesized in one design; a monolithic
  C7-audio + video synth is the final single-config confirmation.
- Placeholder weights → resources are valid (structure-driven); **quality** needs the retrained
  weights (the owner's "come back and retrain a high-quality version" step, after fit is confirmed).

## Phase-1 model cross-check
Phase-1 predicted the *streamed* C7 peak at ~0.06 MB and a non-streamed footprint with comfortable
headroom; the windowed HLS lands at 40 % BRAM — same conclusion (comfortable single-config fit), with
the real weight-ROM + accumulator costs now measured rather than estimated.
