# Reference fact: prior partial wins worth building on (don't reinvent)

Validated techniques from the reference project. Reuse the *idea*; the new project re-implements as
needed for its own architecture.

## ✔ Standalone shared conv engine works
Source: `../UNet-AVSE-Vitis/path3/PHASE1_RESULT.md`.
One runtime-parameterized depthwise-separable conv engine = **71.5 K LUT, 17 DSP, 0 BRAM, 276 MHz** at
the worst-case layer shape. The *datapath* is validated; only the auto-sharing mechanism failed (see
[`dead-ends.md`](dead-ends.md)). A hand-built **offset-addressed engine + a single global activation
pool** is the un-tried, now-unblocked route to a static time-multiplexed single bitstream.

## ✔ Math-exact buffer-elimination transforms
Source: `../UNet-AVSE-Vitis/ALGO_OPTIMIZATIONS.md`. Philosophy = **"recompute the address, don't
materialize the buffer."**
- `concat → split-weights` (no physical concat buffer)
- `upsample + conv` fused via index arithmetic (no 2×T intermediate)
- ⚠️ Caveat: one fusion that removed HLS `DATAFLOW` barriers caused non-converging scheduling. Buffer
  elimination must be validated for **HLS convergence**, not just mathematical equivalence.

## ✔ DATAFLOW + 2D-weight-flatten
Source: `../UNet-AVSE-Vitis/PHASE5_PLAN.md`.
Collapsing 4-D weights to 2-D row-major let the entire video encoder merge into one IP (24× HLS-IR
reduction). Essential technique for any monolithic merge.

## ✔ BRAM-reclaim levers that trade against latency, not LUT
Source: `../UNet-AVSE-Vitis/dfx/AUDIO_DEC_FALLBACK_PLAN.md`.
- Lower weight-ROM partition factor (`dim=2 complete` → `cyclic factor=8`) saved ~600 BRAM18 on
  audio_dec.
- URAM offload via `BIND_STORAGE`.
Since latency is plentiful (RTF 0.468), these are cheap wins.
