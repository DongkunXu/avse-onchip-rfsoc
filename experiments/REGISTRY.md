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

| exp_id | date | candidate / arch | est. peak working set | SI-SDR | PESQ-WB | STOI | params | status | notes |
|---|---|---|---|---|---|---|---|---|---|
| _(none yet — Phase 2 not started)_ | | | | | | | | | |

## Reference anchors (for comparison, not experiments)

| name | SI-SDR | PESQ-WB | STOI | note |
|---|---|---|---|---|
| Noisy input | — | — | — | lower bound (fill from data) |
| Teacher (0.37 M FP32) | — | — | — | the distillation target; fill when the ckpt is located |
| Deployed int16 (reference FPGA) | matches FP32 (cos 0.977, SNR +18.1 dB) | — | — | from the old project's N=160 eval |
