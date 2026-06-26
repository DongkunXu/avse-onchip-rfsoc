# Phase 2 — quality vs deployable working-set Pareto

SI-SDR / PESQ / STOI on LRS3 dev vs the **deployable** on-chip activation working set (MB, audio path;
Phase-1-consistent). Anchors are not retrained here.

> **Eval protocol** — rows marked **[full-dev]** are the definitive measurement: `tools/eval_full_dev.py`
> over **all 3327 dev scenes** (per-scene sliding windows, scene-weighted), mirroring the reference harness.
> Rows marked **[val200]** used the noisy 200-window training val (~25 scenes) and over/under-state quality
> (full-dev revises p2-c7-hq from 4.89 → 4.32). Prefer the [full-dev] numbers.

| candidate | SI-SDR (dB) | PESQ-WB | STOI | working set (MB) | params | note |
|---|---:|---:|---:|---:|---:|---|
| Reference U-Net (4-bitstream) | 5.46 | 1.743 | 0.738 | 4.10 | — | deployed baseline (cited); does NOT fit single-config (215% BRAM) |
| C4 Tiled U-Net (= ref quality) | 5.46 | 1.743 | 0.738 | 0.30 | — | tiling is math-identical to the reference; fits single-config |
| **C7 ConvTasNet (mask) (p2-c7-full)** | **5.40** | **1.727** | **0.754** | **0.017** | 308,544 | **[full-dev]** full data, 80ep/early-stop ep23, best ep18 — ≈ reference, fits single-config |
| C7 ConvTasNet (mask) (p2-c7-hq) | 4.32 | 1.673 | 0.729 | 0.017 | 308,544 | [full-dev] 40k-window subset, 20ep |
| C7 ConvTasNetAVSE (mask) (p2-c7-hq) | 4.89 | 1.683 | 0.718 | 0.017 | 308,544 | [val200] best of 20ep / 40000 win |
| C2 StreamingTCNAVSE (mapping) (p2-c2-r1) | 1.12 | 1.478 | 0.672 | 0.033 | 343,616 | [val200] best of 10ep / 10000 win |
| C7 ConvTasNetAVSE (mask) (p2-c7-r1) | 3.79 | 1.565 | 0.690 | 0.017 | 308,544 | [val200] best of 10ep / 10000 win |

Input (noisy mixture) baseline on full-dev: SI-SDR 2.05, PESQ 1.522, STOI 0.713 → p2-c7-full improves it by
**+3.35 dB / +0.20 / +0.04**.

**Reference anchors** (see `REGISTRY.md`; two different eval subsets, not the same as each other): the
Reference U-Net row above (5.46/1.743/0.738) is the **deployed INT16** number on a small **N=496** subset.
The **FP32 teacher** on a full-dev-scale **N=3319** subset is **3.99/1.673/0.741** — the most comparable
anchor to p2-c7-full's N=3327, and p2-c7-full **beats it on all three** (+1.41 dB / +0.054 / +0.013).