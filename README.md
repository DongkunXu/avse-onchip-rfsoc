# AVSE-OnChip-RFSoC

**Fully on-chip, single-configuration Audio-Visual Speech Enhancement on RFSoC (ZU48DR).**

A from-scratch research project: design an AVSE system that fits **entirely on the
programmable logic of one mid-size FPGA in a single static configuration** — no partial
reconfiguration, no bitstream swapping, no time-multiplexing the chip across phases.

> This is a **new project**, developed in its own repository, parallel to (and independent of)
> the finished 4-bitstream deployment in `../UNet-AVSE-Vitis`. That older deployment is
> **reference material only** — its facts are distilled into [`docs/reference/`](docs/reference/)
> so this project never needs to reach back into it. See [`docs/CHARTER.md`](docs/CHARTER.md).

---

## The goal, in one breath

The reference AVSE U-Net does **not** fit on-chip: its four IPs together need **~215 % of the
chip's BRAM**, so they are loaded as 4 sequential bitstreams via PCAP. The wall is **activations
(≈92.5 % of on-chip memory), not weights**. This project closes that >2× gap by re-architecting
the **dataflow and on-chip memory hierarchy** so a *whole* AVSE pipeline co-resides.

### Priority order (hard constraints, from the project owner)

1. **FIT is the hard constraint.** The entire system must fit on-chip, single-config. If it does
   not fit, nothing else matters.
2. **The contribution is the HARDWARE ARCHITECTURE** (dataflow / memory hierarchy / data movement).
   Publication target is a **circuits / hardware journal**. "We shrank the ML model" is a weak
   contribution; "we designed an on-chip dataflow that fits a complete AVSE in one static config" is
   the paper.
3. **Quality is secondary, maximized within the fit constraint.** Speech-enhancement quality may be
   sacrificed substantially — but never trade *fits* for *sounds better*.

---

## Repository layout

| Path | Purpose |
|---|---|
| [`docs/`](docs/) | Documentation system — charter, roadmap, progress, decisions, inherited reference facts |
| [`analysis/`](analysis/) | **Phase 1**: analytical working-set model (no machine needed) |
| [`src/avse/`](src/avse/) | **Phase 2**: PyTorch — data pipeline, new HW-shaped models, losses, metrics |
| [`experiments/`](experiments/) | **Phase 2**: experiment registry + per-run results |
| [`hls/`](hls/), [`hw/`](hw/) | **Phase 3**: HLS C++ and Vivado/board (finalists only — placeholders for now) |
| [`tools/`](tools/) | Environment setup + standalone utilities |
| `scratch/` | Throwaway exploration (git-ignored) |

## Where to start reading

1. [`docs/CHARTER.md`](docs/CHARTER.md) — the mission and the priority order (read this first)
2. [`docs/reference/bottleneck-diagnosis.md`](docs/reference/bottleneck-diagnosis.md) — *why* a new project is needed (inherit, don't re-derive)
3. [`docs/ROADMAP.md`](docs/ROADMAP.md) — the phased plan and where we are now
4. [`docs/PROGRESS.md`](docs/PROGRESS.md) — the running log of what's done / in progress / next

## Status

**Pre-Phase-1 — scaffolding complete.** Project structure, reference facts, reusable assets, and the
environment are in place. No research code, no training, no synthesis yet. Next concrete deliverable:
the analytical working-set model ([`analysis/`](analysis/)), validated against the reference design's
known numbers, feeding the candidate-architecture scoring table.

## Environment

See [`ENVIRONMENT.md`](ENVIRONMENT.md). TL;DR: Python 3.11 venv (`tools/setup_venv.sh`); torch from the
**cu128** index (RTX 5070 Ti / Blackwell); dataset at `D:\DataSet\LRS3`; Vitis/Vivado 2024.2 for Phase 3.
