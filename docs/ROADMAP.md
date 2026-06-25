# ROADMAP — the phased plan

A **cheapest-filter-first funnel**: broad survey → analytical sieve → cheap quality experiments →
expensive hardware validation on the few that earn it. The rule (owner): **do not write HLS or train
models until the candidate scoring table exists and the owner has picked directions.**

```
 Phase 0          Phase 1               Phase 2                Phase 3
 Diagnosis   →    Analytical      →     PyTorch          →     Hardware
 (DONE)           screening             prototyping            validation
                  (no machine)          (GPU)                  (Vitis/Vivado)
                       │                     │                      │
                  scoring table        Pareto frontier         single-bitstream
                  cut 6-8 → 2-3        quality vs working-set   fit + real reports
                       │                     │                      │
                  ◇ owner picks        ◇ owner picks          ◇ deployment
                    directions           frontier point          engineering
```

---

## Phase 0 — Bottleneck diagnosis ✅ DONE (inherited)

The resource wall is **activations (≈92.5 %), not weights**; root cause is a long, un-tiled time
axis × U-Net skip topology forcing simultaneous residency. Full analysis and the measured numbers
live in [`reference/bottleneck-diagnosis.md`](reference/bottleneck-diagnosis.md). **Inherit these; do
not re-derive.**

## Phase 1 — Analytical screening (no synthesis, no training) ← **WE ARE HERE**

Goal: a defensible, quantitative shortlist *before* spending any GPU or FPGA time.

1. **Build the working-set model** (`analysis/working_set.py`): given an architecture's tensor
   shapes + skip topology + dataflow schedule, compute the peak simultaneously-live activation
   working set (Σ C×T of co-resident tensors), plus rough LUT/DSP/BRAM estimates.
2. **Validate it against the known reference design** (`analysis/validate_baseline.py`): the model
   must reproduce the measured facts — audio_dec ≈ 95 % BRAM standalone and ≈ 215 % BRAM concurrent —
   from the tensor sizes in the diagnosis. *If it can't predict the known answer, it can't be trusted
   on unknowns.* This validation gate is mandatory.
3. **Score the candidates** into a table:
   *approach × peak activation × est. resources × quality risk × engineering effort.*
   Candidate axes (combine, don't pick one — see [`CHARTER.md`](CHARTER.md) §3):
   - Representation: streaming TCN, recurrent/bounded-RF, STFT-domain mask
   - Dataflow: temporal tiling (+halo), off-chip DDR staging of skips, recompute-vs-store
   - Scheduling: single time-multiplexed engine + global activation pool (static bitstream)
   - Capacity: channel/window/fusion reduction, knowledge distillation
   - Asymmetry: cheap video embedding + audio streaming workhorse
4. **Kill anything that can't fit. Cut 6–8 ideas to 2–3.**

**Exit gate ◇**: owner reviews the table and picks the 2–3 directions to prototype.
**Cost**: hours, no machine.

## Phase 2 — Algorithmic prototyping (PyTorch, no hardware)

For survivors: train small versions on LRS3, measure SI-SDR / PESQ / STOI, produce the
**quality-vs-working-set Pareto frontier.** Prove in software how much quality survives at "small
enough to fit" before any HLS time. Distillation (teacher = the 0.37 M reference) is the clean
quality-for-fit lever if pursued.

**Prerequisite**: an explicit owner call on **time-domain vs STFT** ([`DECISIONS.md`](DECISIONS.md) D-2).
**Exit gate ◇**: owner picks a frontier point.
**Cost**: GPU hours (RTX 5070 Ti on this machine).

## Phase 3 — Hardware validation (Vitis/Vivado, finalists ONLY)

For the top 1–3 frontier points: real HLS C-synth → Vivado place-and-route. Confirm **single-bitstream
fit** and capture the real reports — *these reports are the empirical core of the circuits paper.*
**Cost**: high, but bounded to finalists (~6–30 min/IP csynth, ~2.5 h/bitstream).

---

## Methodology notes (own perspective)

- The inherited funnel is sound and adopted as-is. The one emphasis added here: the **Phase-1
  working-set model is the spine of the whole project**, not a throwaway estimate. It is cheap,
  rigorous, and is exactly the quantitative artifact a circuits paper needs to argue "this is *why*
  it fits." Building and validating it well is the highest-leverage early work.
- Each phase has an explicit **owner gate** before the next, expensive phase begins. We do not
  silently roll forward.
