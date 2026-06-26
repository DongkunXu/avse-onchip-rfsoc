# EXPERIMENT REGISTRY

One row per training run (Phase 2). Keep the **table** here as the index; put per-run detail (config
snapshot, full metrics, notes, checkpoint pointer) under `experiments/<exp_id>/`. Checkpoints
themselves are git-ignored — store a path/pointer, not the blob.

## How to log an experiment

1. Pick an `exp_id`: `p2-<NNN>-<short-slug>` (e.g. `p2-001-streaming-tcn-baseline`).
2. Create `experiments/<exp_id>/` with: `config.yaml` (snapshot), `metrics.json` (final numbers),
   `notes.md` (what/why/observations), and a `checkpoint.txt` pointing to the `.ckpt` location.
3. Add a row below. Tie it to the analysis candidate it implements and the working-set estimate it
   was supposed to hit (so software reality can be checked against the Phase-1 prediction).

## Registry

| exp_id | date | candidate / arch | deployable working set | SI-SDR | PESQ-WB | STOI | params | status | notes |
|---|---|---|---|---|---|---|---|---|---|
| **p2-c7-full** | 2026-06-27 | C7 Conv-TasNet (mask) | 0.017 MB | **+5.40** | **1.727** | **0.754** | 308,544 | **done** | full data (315k win), early-stop ep23 on val total-loss, best=ep18. **Metrics are full-dev (3327 scenes, `tools/eval_full_dev.py`)** — beats the FP32 teacher anchor (3.99/1.673/0.741, N=3319) on all three at 1/240 the working set, single-config. Next: export best.pt → HLS ROMs. |
| p2-c7-hq | 2026-06-25 | C7 Conv-TasNet (mask) | 0.017 MB | **+4.89** (best, ep16) | **1.683** | **0.718** | 308,544 | done | 20ep/40k win, cosine LR; final ep19 +4.79; **> reference FP32 (+3.99)** at 1/240 the working set. Next: full-data run. |
| p2-c7-r1 | 2026-06-25 | C7 Conv-TasNet (mask) | 0.017 MB | +3.79 | 1.565 | 0.690 | 308,544 | done | 10ep/10k win (first run) |
| p2-c2-r1 | 2026-06-25 | C2 streaming-TCN (mapping) | 0.033 MB | +1.12 | 1.478 | 0.672 | 343,616 | done | 10ep/10k win; dominated by C7 → masking > direct mapping |

## Reference anchors (for comparison, not experiments)

| name | SI-SDR | PESQ-WB | STOI | note |
|---|---|---|---|---|
| Reference FP32 AV (PAPER_DATA §B, N=3319) | +3.99 | 1.673 | 0.741 | the teacher's quality; C7 already ≈ this after a small run |
| Deployed INT16 AV (PAPER_DATA §B, N=496) | +5.46 | 1.743 | 0.738 | reference FPGA; working set 4.1 MB, does NOT fit single-config |

> Caveat: anchor metrics are from PAPER_DATA on different eval subsets than the p2-* runs (dev-160).
> The comparison is indicative for screening, not a controlled head-to-head. C7/C2 are small runs
> (10 epochs / 10k windows) and still improving — their quality is a lower bound.
