# Phase 3b — Real-weight deployment & deployment-accurate quality (C7)

**Goal (owner):** the *actual* speech-enhancement quality of the **on-chip int16 deployment** of C7
(`p2-c7-full/best.pt`), i.e. the number that is directly comparable to the reference's *deployed-INT16*
anchor (5.46 / 1.743 / 0.738). The Phase-2 number **5.40 / 1.727 / 0.754** is FP32 PyTorch (with the
deployment normalization). This phase produces the fixed-point number **and** wires the real weights into
the HLS design so silicon computes exactly what we measured.

This is the "◇ deployment engineering" gate at the end of the ROADMAP funnel. Method follows the project
spine (Phase 1): **build a model of the deployed compute, validate it against ground truth, then run it at
scale.** Here the "model" is a bit-faithful fixed-point *emulator* of the HLS design; the "ground truth" is
HLS C-simulation.

---

## Two deliverables (decoupled: number first, silicon second)

- **A — the number.** A fixed-point emulator that mirrors the HLS compute (folded BN, hardsigmoid, ap_fixed
  rounding/saturation), built from one exported weight set, run over the full dev split → deployment-accurate
  SI-SDR / PESQ / STOI. No synthesis needed; safe, fast, reversible.
- **B — the silicon.** Make the HLS source *value-faithful* (it is currently a fit-cost proxy with
  placeholder weights, D-9), load the real weight ROMs, C-sim a few windows to prove **emulator ≡ HLS**, then
  re-run csynth + Vivado P&R to confirm the fit still holds with the real model.

`hardsigmoid` (HW mask) is the one genuine design knob; it is kept as the realistic HW choice and its cost
vs `sigmoid` is reported as a sensitivity, not hidden.

---

## Fixed-point formats (authoritative, from `hls/src/c7_types.hpp`)

| type | format | int/frac | range | step | rounding / overflow |
|---|---|---|---|---|---|
| `sample_t` | `ap_fixed<16,1>` | 1 / 15 | [-1, 1-2⁻¹⁵] | 2⁻¹⁵ | TRN, SAT |
| `data_t` (activations) | `ap_fixed<16,7>` | 7 / 9 | [-64, 64-2⁻⁹] | 2⁻⁹ | TRN, SAT |
| `wgt_t` (weights) | `ap_fixed<16,5>` | 5 / 11 | [-16, 16) | 2⁻¹¹ | TRN, WRAP |
| `acc_t` (accumulator) | `ap_fixed<48,22>` | 22 / 26 | ±2²¹ | 2⁻²⁶ | TRN, SAT |

- **TRN** = truncate toward −∞ = `floor(x/step)*step`. **SAT** = clamp to [min,max]. **WRAP** = modulo (weights
  are all |·|<2 ≪ 16, so wrap never triggers → treat as exact-to-step).
- `acc_t` is wide enough (26 frac bits, ±2²¹) that within a single conv/matmul accumulation it is effectively
  exact relative to the 16-bit operands; quantization to `data_t` happens only on **buffer writes**.
- Emulator quantizers: `q_act(x)=clamp(floor(x·512)/512,-64,64-2⁻⁹)`, `q_wgt(x)=floor(x·2048)/2048`,
  `q_io(x)=clamp(floor(x·32768)/32768,-1,1-2⁻¹⁵)`. Quantize at exactly the points HLS writes a buffer.

---

## Deploy compute graph (single source of truth) — PyTorch op → deploy math → HLS array

Channels: N=128 (enc/latent), B=64 (bottleneck), H=128 (TCN), NBLK=10, KD=3, L=32, STRIDE=16,
**T_LAT = 1201** (not 1200 — see gap G1), TF=30 video frames, vid C0=64, C=96.

### Audio mask network (`c7_audio_core.hpp`)
1. **encoder** `Conv1d(1→N,k=L,s=STRIDE,pad=STRIDE,bias=False)` → `w[N][T_LAT]`, quant `data_t`.
   index: `s = t·STRIDE + k − STRIDE`, valid `0≤s<T`. ROM `Wenc[N][L] = encoder.weight[n,0,k]`.
2. **in_norm fold + bottleneck + video** → `y[B][T_LAT]`:
   `wn = bn_s[n]·w[n] + bn_b[n]` (in_norm applied inline, exact), `y[b] = Σ_n wn·Wbn[n][b] + video_embed[b]`.
   `bn_s[n]=γ/√(σ²+ε)`, `bn_b[n]=β−bn_s·μ` (from `in_norm`), `Wbn[n][b]=bottleneck.weight[b,n,0]`.
3. **10 dilated dwsep TCN blocks** (residual), block i, dil=2^(i mod 5):
   - IN1x1: `h[c] = bn1( prelu(Σ_b y[b]·Win[i][b][c], pr1[i][c]) )` quant `data_t`.
     `Win[i][b][c]=tcn.i.in_conv.weight[c,b,0]`, `pr1=tcn.i.prelu1.weight`. **bn1 kept inline** (per-channel
     affine `bn1_s[c]·x+bn1_b[c]`, like in_norm), NOT folded — see note below.
   - DW: `hd[c] = bn2( prelu(Σ_j h[c][t−(KD−1−j)·dil]·Wdw[i][c][j], pr2[i][c]) )` quant `data_t`.
     `Wdw[i][c][j]=tcn.i.dwconv.weight[c,0,j]`, `pr2=tcn.i.prelu2.weight`, bn2 inline. (h zero-padded on the
     left for `tt<0`.)
   - OUT1x1: `y[b] += Σ_c hd[c]·Wout[i][c][b]`, `Wout[i][c][b]=tcn.i.out_conv.weight[b,c,0]`.

   > **Why bn1/bn2 are inline, not folded:** PyTorch left-pads the bn1 *output* with **zeros**
   > (`F.pad(bn1(...), (pad,0))`), so the padded taps carry 0, not `bn1(0)=bn1_b`. Folding bn1 into the
   > dwconv as a single per-channel bias would add `bn1_b·Σ(taps)` at every output, wrong by the pad
   > contribution on the first `(KD−1)·dil` columns of each block. Applying bn1/bn2 as an explicit
   > per-channel affine on the stored `data_t` buffer (then zero-padding) is exact and costs the same as the
   > already-inline in_norm. `bn_s=γ/√(σ²+ε)`, `bn_b=β−bn_s·μ`, ε=1e-5.
4. **mask** `Wmask[b][n]=mask_conv.weight[n,b,0]`: `w[n] = w[n]·hardsigmoid(Σ_b y[b]·Wmask[b][n])`.
   `hardsigmoid(x)=clamp(0.2x+0.5,0,1)`.
5. **decoder** `ConvTranspose1d(N→1,k=L,s=STRIDE,pad=STRIDE,bias=False)`:
   `out[s] = Σ_{n,k} w[n][t]·Wdec[n][k]` where `s = t·STRIDE + k − STRIDE`, valid `0≤s<T`. `Wdec[n][k]=decoder.weight[n,0,k]`.

### Video encoder (`c7_video.hpp`) + conditioning (`c7_avse_top.cpp`)
Per frame (30): `Conv2d(1→64,k7,s2,p3)+BN+ReLU` → 3× `DepthwiseSeparableConv2d(stride2)` (depthwise(k3) →
pointwise → BN → **ReLU then + shortcut[Conv1x1 s2 + BN]**) → `AvgPool2d(k5,s1)` (6×6→2×2) →
`feature_proj Conv2d(k2)+bias+ReLU` (2×2→1×1) → `temporal_proj Linear+bias + residual`. → `video_feat[96][30]`.
Conditioning: `proj Conv1d(96→64,1)+bias` per frame, then nearest upsample 30→T_LAT
(`frame = floor(t·TF/T_LAT)`) → `video_embed[B][T_LAT]`. All BN folds into its preceding conv as a per-channel
bias; quantize each conv/BN/ReLU output to `data_t`.

---

## Fidelity gaps (HLS as-is → faithful) — every one must be closed for B

| # | gap | current HLS | faithful fix |
|---|---|---|---|
| G1 | latent length | `T_LAT=1200` | **1201** (enc/dec/buffers/loops) |
| G2 | decoder offset | `s=t·STRIDE+k` (16-sample shift) | `s=t·STRIDE+k−STRIDE` |
| G3 | TCN bn1/bn2 | dropped | keep **inline** per-channel affine `bn1_s/b[H]`, `bn2_s/b[H]` (not folded — zero-pad boundary) |
| G4 | mask | hardsigmoid (keep) | keep; report sigmoid sensitivity |
| G5 | video conv0 BN | none | fold into conv0 bias |
| G6 | DWSep BN+shortcut | none, extra post-depthwise ReLU | depthwise(no relu)→pointwise→fold-BN→ReLU→+shortcut(1×1 s2 + fold-BN) |
| G7 | video pool/head | global mean 6×6→1 | AvgPool(k5,s1)→2×2 then feature_proj Conv2d(k2)+bias+ReLU |
| G8 | temporal_proj bias | none | add `bias_tp[C]` |
| G9 | video proj bias | none | add `bias_vproj[B]` |
| G10| upsample index | `t/40` (breaks at 1201) | `floor(t·TF/T_LAT)` |
| G11| all weights | index-seeded placeholder | real exported ROMs |

For deliverable **A** (emulator) all of G1–G11 are modelled directly from `best.pt`; only **B** edits the HLS.

---

## Steps & status

1. ✅ This reconciliation spec (the deploy compute graph + gaps).
2. ⏭ `tools/export_weights.py` — `best.pt` → fold all BN → quantize → emit `c7_weights.npz` (emulator) +
   `hls/src/c7_weights.hpp` (HLS ROMs). One weight truth source. Verify folded-fp model ≡ original (fp tol).
3. ⏭ `tools/c7_deploy.py` — bit-faithful fixed-point emulator (audio_core + video_encoder) from the npz.
   Gate: emulator at full precision ≡ PyTorch `best.pt` (proves the reimplementation+fold); int16 path runs.
4. ⏭ Deployment-accurate eval over **full dev** (reuse `eval_full_dev` protocol) → the number; report the
   int16-vs-FP32 quantization cost and the hardsigmoid-vs-sigmoid sensitivity.
5. ⏭ Make HLS value-faithful (G1–G11) + load ROMs; C-sim a few windows → **emulator ≡ HLS** trust gate.
6. ⏭ Re-run csynth + Vivado P&R with real weights → final resource numbers (confirm fit holds).
7. ⏭ Update PROGRESS / REGISTRY / DECISIONS / memory; commit each part.

**Op note:** training is done, GPU free; P&R (step 6) is heavy — run alone (D-11).
