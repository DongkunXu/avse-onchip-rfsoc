# Phase 3 plan — HLS fit check for C7 (owner: fit-first)

Owner directive (DECISIONS D-9): **first confirm C7's structure actually fits** via real Vitis HLS
C-synth + Vivado synth/P&R reports, **then** iterate / retrain a high-quality version. Fit is
structure-driven, so we synthesize with **placeholder weights** — decoupled from quality.

## Scope 3a — the audio mask network (the part that decides whether the new architecture fits)

The reference's wall was the **audio** path (skip residency). C7 removes it. So 3a synthesizes C7's
audio path and checks its real resource use; the video encoder is a separate, known-cheap IP (reference
~38% BRAM) whose embedding is an **input port** here (single-config: video IP feeds audio IP on-chip).

Pipeline (windowed, int16 `ap_fixed<16,7>`):
```
audio[19200] --enc(Conv1d 1->128, k32 s16)--> w[128][1200]
video_embed[64][1200] (input)                 (conditioning)
y = bottleneck(BN(w)) + video      [64][1200]
for 10 dilated dwsep TCN blocks:   y += out1x1(dwconv(in1x1(y)))
mask = sigmoid(mask1x1(y));  w *= mask
out[19200] = decoder(ConvTranspose 128->1, k32 s16)(w)
```
On-chip buffers (minimised by reuse): w[128][1200], y[64][1200], h[128][1200], o[64][1200]
≈ 0.46 M int16 ≈ ~450 RAMB18 ≈ ~21% BRAM — Phase-1 predicts comfortable fit, with headroom for video.

## Steps
1. `hls/src/c7_types.hpp` — int16 types (from the reference). ✅
2. `hls/src/c7_audio_top.cpp` — the audio mask network, synthesizable, placeholder weights.
3. `hls/tcl/run_csynth_c7.tcl` — Vitis HLS 2022.2 csynth, target xczu48dr-ffvg1517-2-e.
4. Read `csynth.rpt` → BRAM/DSP/LUT estimate; compare to the Phase-1 prediction (re-validate the model
   on the NEW architecture).
5. Vivado synth + P&R → the real utilization report (the empirical core of the fit claim).

## Success = the report shows C7's audio path well under budget, leaving room for the video IP, i.e.
the whole AVSE can co-reside in ONE static configuration — the thing the reference could not do.
