# Reference fact: settled dead-ends — DO NOT REPEAT

Each was tried and documented on the reference project. Re-trying them burns time for no gain.

## ⛔ INT8 / DPU
Source: `../UNet-AVSE-Vitis/dpu_trial/CONCLUSION.md`.
INT8 PTQ costs **−1.6 to −2.9 dB SI-SDR** — breaks the quality floor. SI-SDR is log-sensitive to the
quantization noise floor (silence / noise-floor regions). QAT judged unlikely to recover enough.
**Precision is locked at int16** — which is *why* 16-bit activation buffers dominate BRAM. Do not try
to shrink by going to 8-bit.

## ⛔ AI Engine (AIE) offload
Source: `../UNet-AVSE-Vitis/aie_hybrid/`.
The ZU48DR / RFSoC 4x2 **has no AI Engines** (Versal-only). That workspace is secretly a *board-change*
proposal to VCK190 and was never built. Irrelevant unless the owner agrees to change hardware.

## ⛔ Letting HLS implicitly time-multiplex one engine across calls
Source: `../UNet-AVSE-Vitis/path3/PHASE2_FINDING.md`.
Constant-propagation + array-partition mismatch makes Vitis emit **N specialized copies** → zero
saving. Real sharing needs a **hand-built memory-pool / offset-addressed engine**, not a hope that HLS
will share automatically.

## ⛔ Naive per-IP shared compute pool
Makes BRAM **worse** (130 % → 187 %) — fatal when BRAM is the binding constraint. A shared engine only
helps if paired with a *single global activation pool* (see [`prior-wins.md`](prior-wins.md)), not a
per-IP pool.

---

## What CHANGED since these explorations (these unblock new options)

- **196 GB RAM on the authoring machine** (was 32 GB) retired the integrated-`avse_top` csynth OOM that
  killed the single-engine global-pool merge and DFX PR_VERIFY. A monolithic single-bitstream synthesis
  is now tractable. *(Note: verify available RAM on the current machine before relying on this.)*
- **Latency slack** — compute RTF 0.468 means **~half the time budget is free.** Trade latency for BRAM
  (lower partition factors, serialized streaming, DDR round-trips with double-buffering) nearly for free.
- **URAM 78 % unused** — only audio_dec touches it (36/80). Significant untapped on-chip memory *if*
  16-bit data is packed into URAM's 72-bit words (4×16-bit per word fixes the naive-packing waste).
- **DDR4 essentially unused by the compute path** — GBs available for off-chip activation staging.
