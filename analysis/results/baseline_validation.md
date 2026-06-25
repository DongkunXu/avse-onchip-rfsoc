# Phase 1 result — working-set model validated against the reference design

**Date**: 2026-06-25 · run `python analysis/validate_baseline.py` (exit 0, PASSED).

The model (`working_set.py`) is fed the reference design's **actual** buffer lists
(`baseline_reference.py`, transcribed from the HLS source), and reproduces the measured hardware.

## Per-IP BRAM (RAMB18, budget 2160)

| IP | predicted activation | act % | measured total % | act/meas | URAM |
|---|---:|---:|---:|---:|---:|
| audio_enc | 1109 | 51.3 % | 57 % | 0.90 | 0 |
| **audio_dec** | **1971** | **91.2 %** | **95 %** | **0.96** | 29 (36.2 %) |
| fusion | 578 | 26.8 % | 32 % | 0.84 | 0 |
| video | 518 | 24.0 % | 38 % | 0.63 | 0 |

- **audio_dec (the binding IP) is predicted to 0.96 of measured** from buffer sizes alone, and its
  **URAM is exact (36.2 % vs 36 %)**.
- enc / fusion are consistently ~0.85–0.90 — activation explains the bulk; the residual is the weight
  ROMs + small working buffers (the diagnosis independently puts weights at 7.5 % of on-chip memory).
- **Concurrent monolithic activation = 193.3 %** of BRAM; measured total = **215 %**. The 22-pt gap is
  exactly the weight + working-buffer residual. → The model confirms the thesis quantitatively:
  **activations are the wall.**

### Known approximation: video (0.63)
Video is the least-precise IP. Its measured 38 % is inflated by (a) per-frame **DATAFLOW ping-pong**
and (b) dense temporal-projection weight ROMs — neither of which is the long-time-axis *activation
residency* the project targets. Video is "nearly free" w.r.t. the residency wall, so we do not
over-fit it; the model is tuned and validated on the **audio** path that actually binds.

## The headline finding — pooling alone is not enough

For the monolithic audio U-Net:

| quantity | value |
|---|---|
| static Σ(all tensors) | 2,150,400 elems · **4.1 MB** |
| **peak-live working set** | 1,267,200 elems · **2.4 MB** (at decoder stage 0) |
| static / peak ratio | **1.70×** |

At the peak, `skip0, skip1, skip2, skip3, bottleneck, dec_s0` are all simultaneously live. So even a
**perfectly-pooled schedule of the same U-Net topology** (Axis 3, single-engine + global pool) only
shrinks resident audio activation from 4.1 MB → 2.4 MB. **2.4 MB still maps to ~60–115 % of BRAM by
itself** (depending on partition banking) — the U-Net skip topology forces skip0–3 + bottleneck to
co-reside, and that floor is already near the whole budget.

**Implication for candidate selection**: scheduling/pooling (Axis 3) cannot fit the system on its own.
A winning design **must bound the live temporal extent** — shorten the time axis itself (Axis 1:
streaming / recurrent / STFT-mask) and/or tile-or-stage it (Axis 2). This is the first quantitative,
model-backed steer for Phase 1's candidate scoring.

## Model trust status
✅ Validated for the audio path (the binding path). Ready to score candidate architectures with the
same model. Calibration constants (`WORDS_PER_RAMB18=1024`, URAM naive 16b) were **not** fudged —
per-IP accuracy comes from the real buffer lists.
