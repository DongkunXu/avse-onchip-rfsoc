# Reference fact: the bottleneck diagnosis (inherit, do not re-derive)

Measured on the reference 4-bitstream deployment. This is **Phase 0**, already done. These numbers
are the ground truth the Phase-1 analytical model must reproduce.

## The one-line takeaway

> This is a **temporal-axis / dataflow problem masquerading as a BRAM problem.** The model doesn't fit
> because it is a time-domain waveform U-Net that holds a 19200-long signal at multiple resolutions
> plus all skips at once. Fixing it means bounding the **live temporal extent** of activations — by
> representation, by dataflow, or both — **not** by shrinking weights.

## 1. The wall is ACTIVATIONS, not weights

| Category | Whole-design footprint | Share |
|---|---|---|
| Weights (all parameter ROMs) | ~695 KB | 7.5 % |
| **Activations** (feature-map / skip / bottleneck staging) | **~8.4 MB** | **92.5 %** |

Activations are ~12× the weights. The model is tiny (0.37 M params ≈ 0.75 MB int16).
**Compressing/quantizing/pruning weights will NOT solve this.**

## 2. Measured concurrent resource use (all 4 IPs + PS, post-synthesis)

Source: `../UNet-AVSE-Vitis/dfx/vivado_proj/util_concurrent_synth.txt`

| Resource | Used | Available | Util |
|---|---|---|---|
| **BRAM (RAMB36)** | 2327 | 1080 | **215 %** ← the wall |
| **LUT** | 536,722 | 425,280 | **126 %** |
| DSP | 1322 | 4272 | 31 % |
| FF | 442,567 | 850,560 | 52 % |
| URAM | 0 | 80 | 0 % |

Implementation never completed — placement is physically impossible at 215 % BRAM. **This is the hard
evidence for the 4-way split.** Standalone, the binding IP is **audio_dec at 95.4 % BRAM** by itself
(and only fits via a URAM offload of one buffer). Shared PS + AXI infra costs **0 BRAM** — 100 % of
the pressure is the neural-network logic.

## 3. Root cause: long time axis × U-Net skip topology

A U-Net keeps **every encoder skip alive from production until its mirror decoder stage consumes it.**
The decoder consumes skips in reverse, so the longest-lived skip is held across the entire
encode → fuse → decode pass. The simultaneously-resident set:

| Tensor | C × T | Elements | int16 bytes |
|---|---|---|---|
| **skip1 (enc0 out)** | **32 × 9600** | 307,200 | 600 KB |
| **skip2 (enc1 out)** | **64 × 4800** | 307,200 | 600 KB |
| skip3 (enc2 out) | 96 × 2400 | 230,400 | 450 KB |
| skip4 (enc3 out) | 128 × 1200 | 153,600 | 300 KB |
| bottleneck (enc4 / fusion) | 192 × 600 | 115,200 | 225 KB |
| **must co-reside** | | **1,113,600** | **~2.2 MB** |

**Counter-intuitive but decisive**: the *shallowest* skips (fewest channels) are the **most
expensive**, because activation cost ∝ C × T and the audio time axis runs 19200 → 9600 → 4800 → 2400
→ 1200 → 600. T halves per layer while C grows only ~1.3–2×, so the C×T product peaks at the shallow
end. The two shallowest skips alone are 55 % of the resident activation.

Three properties combine to force this (in order of impact):
1. **Long, un-tiled time axis** — the full 19200-sample window is materialized at every resolution.
2. **U-Net skip topology forces simultaneous residency** — store-now-use-much-later.
3. **No streaming/dataflow across the U-Net** — decode depends on full encode+fusion.

The **video path is nearly free** (collapses spatial dims to ~1×1 early; only ~30×96 features). So the
whole problem is the **audio time axis** → "audio-visual asymmetry" is a natural design lever.

## 4. What this implies for design

Minimize **peak Σ C×T over co-resident tensors**. Bound the live temporal extent so peak activation
**decouples from window length**. Levers: representation (streaming / recurrent / STFT-mask),
dataflow (tiling + halo / DDR staging of skips / recompute-vs-store), scheduling (one time-multiplexed
engine + global activation pool). See [`../ROADMAP.md`](../ROADMAP.md) and [`prior-wins.md`](prior-wins.md).
