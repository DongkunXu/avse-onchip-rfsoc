# Phase 2 plan — PyTorch prototyping → quality-vs-working-set Pareto

Goal: put **quality** (SI-SDR / PESQ / STOI on LRS3) on the same axis as the **working set** (from the
validated Phase-1 model), for the chosen time-domain candidates, before any HLS. Owner picks the
Pareto operating point afterward (gate after Phase 2).

## What gets trained vs. anchored

| cand | what | quality source |
|---|---|---|
| **C4** Tiled U-Net | reference U-Net run in temporal tiles (math-identical w/ halo) | **anchored** to the reference's measured quality (PAPER_DATA: SI-SDR ~+4.0 AV FP32). Not retrained — tiling doesn't change the function. (Teacher ckpt not on this machine; quality is documented.) |
| **C2** Streaming TCN | dilated causal TCN audio backbone + video cond. | **train small** |
| **C7** Conv-TasNet-style | encoder → TCN mask → decoder, no U-Net skips + video cond. | **train small** (headline) |

Levers C8/C9 are schedule/representation tweaks on a base — evaluated only if a base needs headroom.

## Method

1. **Implement** C7 and C2 as AVSE models (reuse the validated, cheap video encoder; new audio
   backbone). Verify forward pass + param count on a real batch. ✅ first.
2. **Working-set instrumentation**: for each model, compute its peak-live activation working set
   (same metric as Phase 1) from its layer shapes, so the trained quality maps onto the Phase-1 axis.
3. **Train small** versions on LRS3 (dev for quick iteration; train split for the real runs). Same
   loss (migrated `ImprovedAVSELoss`) and metrics (`avse.metrics`). Log to `experiments/<exp_id>/`.
4. **Evaluate** SI-SDR / PESQ / STOI on a held-out set; record in `experiments/REGISTRY.md`.
5. **Pareto**: plot quality vs peak-live working set; bring to owner.

## Practical
- GPU: RTX 5070 Ti (16 GB), torch 2.11+cu128. Start with short runs (few epochs, subset) to validate
  the harness + get rough quality, then scale.
- Keep models **HW-aware** (depthwise-separable convs, bounded ops, int16-friendly ranges) since the
  winner goes to HLS in Phase 3 — but Phase 2 optimises quality; HW-friendliness is structural.
