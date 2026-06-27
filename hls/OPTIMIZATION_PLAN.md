# Phase 4 — Throughput / performance optimization (the deferred D-19 phase)

**Owner-authorized (2026-06-27):** now that the whole AVSE is proven end-to-end on real silicon
(single static bitstream, on-board SI-SDR +6.66), optimize throughput while finding the
**resource × efficiency balance**. No shortcuts on the computation — every optimization must stay
**value-faithful** (C-sim ≡ fixed-point emulator, `rel_rms < 1e-2`) and, where it touches a
scatter/accumulate or pipeline hazard, be re-checked **on-board** (C-sim cannot catch RMW hazards).

This document is the plan-of-record and progress tracker for the phase. Follow
[[working-principles]] (plan first, no shortcuts, root-cause) and [[autonomy-authorization]]
(self-drive; commit + doc after each part).

---

## Baseline (measured, current `master` — rolled video, real weights)

Monolithic `c7_avse_top` csynth (`hls/build/c7_avse`, 2022.2, xczu48dr, 5 ns / 200 MHz):

**Total latency = 512,848,453 cycles = 2.564 s/window** (1 window = 1.2 s audio @ 16 kHz).
**On-board measured = 11.67 s/window** (4.55× the csynth estimate — un-bursted `video_in` DDR reads).

| Module | cycles | time | % total | why |
|---|---:|---:|---:|---|
| **video_encoder** | 443.1M | 2.215 s | **86.4 %** | serialized reductions |
| · CONV0 (1→64, k7, s2) | 216.8M | 1.084 s | **42.3 %** | II=49, 49-MAC kernel not unrolled |
| · PW1+SC1 (→96, 1×1) | 106.2M | 0.531 s | 20.7 % | II≈32, 64-ch reduction serialized |
| · PW2+SC2 (96→96, 1×1) | 88.4M | 0.442 s | 17.2 % | II≈32, 96-ch reduction serialized |
| · PW3+SC3 (96→96, 1×1) | 22.1M | 0.110 s | 4.3 % | smaller spatial (6×6) |
| · DW1/2/3, POOL, FP, TPROJ | 9.6M | 0.048 s | 1.9 % | small |
| **audio_core** | 69.5M | 0.347 s | 13.5 % | |
| · BLOCKS (10 TCN) | 52.3M | 0.261 s | 10.2 % | IN1x1+OUT1x1 II≈16–32 |
| ·· IN1x1 (B→H 1×1) ×10 | 24.6M | 0.123 s | 4.8 % | 64-ch reduction, partition factor 2 |
| ·· OUT1x1 (H→B 1×1) ×10 | 24.6M | 0.123 s | 4.8 % | 128-ch reduction, partition factor 2 |
| ·· DW (depthwise k3) ×10 | 3.1M | 0.015 s | 0.6 % | already II=2 |
| · ENC (Conv1d 1→128) | 4.9M | 0.025 s | 1.0 % | II=32 |
| · DEC (ConvT, rolled) | 7.4M | 0.037 s | 1.4 % | rolled (hazard fix); gather rewrite later |
| · BOT / MASK | 4.9M | 0.025 s | 1.0 % | |
| VPROJ + VUP | 0.28M | ~0 | 0.05 % | negligible |

Resource (post-route, real weights): **BRAM 83 % (binding), LUT 20 %, DSP 17 %, FF 5 %, 200 MHz met.**
→ ~17 % BRAM headroom; huge DSP/LUT headroom. Spend DSP/LUT for latency; guard BRAM.

**Correction to D-19:** D-19 claimed "the audio path dominates latency" — the report shows the
**video encoder is 86 % of latency**. Rolling the video was right (it escaped the 6 h synth blow-up
and got us to bitstream), but the latency justification was wrong. The video is the #1 target.

---

## Targets

- **Milestone 1 — real-time:** < 1.2 s/window (currently 2.564 s csynth / 11.67 s board).
- **Milestone 2 — beyond:** maximize throughput within the BRAM budget; report the
  resource × latency Pareto (the circuits-architecture contribution).
- **Hard invariant:** value-faithful (C-sim PASS) at every step; on-board re-validation before claiming a board number.

---

## Plan (incremental, measured, one loop at a time — to catch any synth blow-up on a single 5-min run)

Work in a **separate** build project (`c7_avse_opt`) so the baseline `c7_avse` stays intact for A/B.

- **O-1 — on-chip frame caching (DDR fix).** Burst-load each 96×96 frame into a local BRAM buffer
  once/frame; CONV0 reads on-chip, not DDR. Value-identical. Fixes the 4.55× board penalty + likely
  the ~2 % residual. Prereq for pipelining the video convs. *Gate: C-sim PASS; csynth sanity.*
- **O-2 — parallelize the video compute (incremental).** One loop at a time, csynth + C-sim after each:
  - O-2a CONV0: unroll the 7×7 kernel reduction (≈49 MAC/clk) → II 49→~1. ~50× on CONV0.
  - O-2b PW1/SC1: partition the channel-reduction (modest factor), unroll; **avoid** complete-
    partitioning the weight ROMs (that caused the 6 h blow-up). Restructure strided shortcut reads
    if bank-conflict analysis explodes.
  - O-2c PW2/SC2, PW3/SC3, DW1/2/3, POOL, FP, TPROJ.
  - Consider **dataflow across the 30 frames** (coarse pipeline w/ ping-pong buffers) if per-loop
    unroll alone doesn't reach target — overlaps frame f+1's CONV0 with frame f's tail.
- **O-3 — audio TCN pointwise convs.** Raise partition factor on y/h/hd + unroll the B/H reductions
  → IN1x1/OUT1x1 II 16–32 → 1–4. Guard BRAM. Optionally gather-rewrite DEC.
- **O-4 — (optional) module reuse / dataflow** if a better resource×latency point exists.
- **O-5 — integrate → P&R → bitstream → on-board.** Full csynth → Vivado P&R (run alone, D-11) →
  bitstream → board; measure on-board latency + quality on a larger subset. Update docs/REGISTRY/
  DECISIONS (supersede D-19's latency claim) / memory.

## Log
- _(2026-06-27)_ Phase opened. Baseline captured above. Starting O-1.
- _(2026-06-27)_ **O-1 done** (frame cache). C-sim PASS (worst rel_rms 8.54e-3, value-identical).
  csynth `c7_avse_opt`: **512.85M → 406.96M cyc (2.564 → 2.035 s, −20.6%)**. CONV0 alone
  7.225M → 3.686M/frame (II 49→25): staging to 2-port BRAM vs the single-port `m_axi` model
  nearly halved it. BRAM 72% est (+~14 for fbuf), DSP 19%. On-board: frame now read contiguously
  (LOADF burst) vs CONV0's old strided per-pixel DDR reads. TODO: bump `max_read_burst_length`.
  Next: O-2a (unroll CONV0 kernel → II≈1).
- _(2026-06-27)_ **O-2a done** (unroll CONV0 7×7 kernel). C-sim PASS (rel_rms 8.54e-3, unchanged).
  fbuf→2D partitioned cyclic 7×7 (49 banks); conv0 weights staged to a fully-partitioned local
  buffer; ox loop pipelined II=1 with the kernel unrolled. **CONV0 II 25→1**: 3.686M → 0.147M
  cyc/frame. Total **406.96M → 300.79M cyc (2.035 → 1.504 s)**, −41% vs baseline. BRAM 74% (+35),
  DSP 22% (+135). Pointwise/shortcut convs now 72% of the whole design (216.5M) → O-2b is next.
- **O-2b approach (decided):** partition the reduction-input activation buffers (dw, b0, b1, b2) on
  the *channel* dim (cyclic factor 16, keeps BRAM ~flat — complete would double it via bank rounding);
  load each output channel's weight *row* into a complete-partitioned register array (LUT/FF, cheap)
  and unroll the channel reduction → II≈4 (Cin 64) / 6 (Cin 96). Channel→bank, spatial→address makes
  the strided shortcut reads conflict-free (the original 6 h blow-up was strided-into-spatial +
  complete-partitioned big ROMs — both avoided). Do PW loops first, then the strided SC loops.
