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

## Training-run roadmap (owner, 2026-06-25)

1. **`p2-c7-hq` (running)** — C7, **40 000-window subset** of the LRS3 train split (full=315 253
   windows), **data_mode=full** (both noise + competing-speech interferers, NOT noise-only), 20 epochs,
   cosine LR. First HQ run — observe how far quality goes. (At ep2 already > the 10-epoch run.)
2. **THEN a full-data run** — after `p2-c7-hq` finishes, retrain on the **whole** train split
   (`--max-train-windows 0` → 315 253 windows). ~8× longer per epoch; the definitive quality model +
   real-weight export for Phase 3.

## Hardware tuning to squeeze this machine (do AFTER the current run; owner request)

Goal: fully use the hardware without spilling. Measured snapshot during `p2-c7-hq` (batch 16,
workers 3): **VRAM 4.0 / 16.3 GB (≈25 %)**, GPU util 91 %, ~437 s/epoch on 40k windows, no shared-mem
spill. So there is large headroom to exploit:

- **GPU 16 GB VRAM — use it, never overflow to shared.** Spilling to shared memory *runs but is much
  slower*. Raise **batch size** (4 GB at bs=16 → bs≈48–64 should still fit well under 16 GB); keep a
  **~1–2 GB safety margin**. Watch `nvidia-smi memory.used` stays < ~14.5 GB.
- **Feed the GPU**: GPU is already 91 % at bs=16, so the data pipeline may bind at larger batch —
  raise **workers** (e.g. 4–6) and/or **prefetch** so the bigger batches don't starve. Balance against
  the 32 GB system RAM and the in-place window-subset fix (so workers stay light).
- **System RAM 32 GB** (commit ~42 GB): serialize with Vivado (DECISIONS D-11). Full-data (315k
  windows) makes the dataset/window-list bigger — keep workers modest and watch committed memory.
- **Sweep other params** for the best throughput/quality balance: batch, LR (scale with batch), workers,
  prefetch, epochs. Pick the config that maximises GPU utilisation while staying within VRAM + RAM.

> These tunings apply to the **full-data run**; the current `p2-c7-hq` is left untouched to finish.
