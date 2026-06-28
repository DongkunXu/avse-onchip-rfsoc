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

## Status (updated 2026-06-28)

**The whole goal is achieved on real hardware AND optimized: a complete, real-weight, single-static-config
AVSE runs on the RFSoC 4x2, enhances speech, and now runs ~4.2× faster than real-time** (286 ms/window).
End-to-end: full-data training → deployment-accurate quality → value-faithful HLS → P&R → bitstream →
on-board measurement → **throughput optimization (40.8× faster on-board, at lower BRAM).**

- **Phase 1 (analytical) — DONE.** Working-set model (`analysis/`) validated against the reference (reproduces
  its 215 % concurrent BRAM) picked **C7 — a Conv-TasNet-style time-domain mask network with NO U-Net skips**
  (so the skip-residency wall that forced the reference into 4 bitstreams does not exist).
- **Phase 2 (PyTorch) — DONE.** C7 trained on the full data (`p2-c7-full`): **full-dev SI-SDR +5.40 / PESQ
  1.727 / STOI 0.754** (FP32), *above* the FP32 reference (+3.99) at a **0.017 MB** working set (1/240 of the
  reference).
- **Phase 3a (HLS fit, placeholder weights) — DONE.** Established the structure fits one static config.
- **Phase 3b (real-weight deployment) — DONE.**
  - **Deployment-accurate quality** (int16 fixed-point emulator, C-sim-validated): **SI-SDR 4.98 / PESQ 1.63 /
    STOI 0.742** full-dev (int16 quant −0.33 dB; hardsigmoid mask −0.085 dB).
  - **Real-weight monolithic P&R:** **83 % BRAM, 20 % LUT, 17 % DSP, 200 MHz** (one static config;
    `hls/RESULTS_avse_monolithic.md`). vs reference 215 % BRAM / 126 % LUT across 4 PCAP bitstreams.
  - **On-board (RFSoC 4x2, PYNQ), single static bitstream:** **SI-SDR +6.66 / PESQ 1.72 / STOI 0.72** on a
    16-window subset — beats the mixed input by **+2.27 dB** and matches the emulator to **−0.22 dB**
    (corr 0.9855). **The central hypothesis is proven on real silicon, enhancing speech.**

- **Phase 4 (throughput optimization) — DONE** (DECISIONS D-20; `hls/OPTIMIZATION_PLAN.md`). On-chip
  frame/audio caches, unrolled conv/TCN reductions (channel-partition + register weights), and a **gather
  decoder** replacing the rolled scatter — **all C-sim bit-identical** to Phase 3b. **Post-route: BRAM 76.8 %
  (↓ from 85.3 %), DSP 36.6 %, LUT 32.1 %, 200 MHz met.** **On-board 286 ms/window — 40.8× vs the 11.67 s
  baseline → 4.2× under real-time**, quality identical (SI-SDR 6.66, corr 0.9855 = baseline). The optimization
  spent DSP/LUT/FF and *freed* BRAM — the resource×efficiency contribution, on real silicon. The decoder RMW
  hazard is eliminated at the root (gather).

### Open / optional items
- The small silicon-vs-emulator residual (corr 0.9855, quality-negligible) is **pre-existing and unchanged**
  by the optimization (so it is *not* the un-bursted DDR reads, as once hypothesized) — a silicon-vs-C-sim
  effect, a candidate for a focused co-sim investigation.
- **Optional further speedup:** ~23 % BRAM headroom remains → the TCN core (BLOCKS) II→1 is feasible for ~1.2×
  more; held since the design is already well under real-time.
- Operational: 32 GB host — run heavy jobs one at a time (D-11); board access in
  [memory `board-access`]; full timeline in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Environment

See [`ENVIRONMENT.md`](ENVIRONMENT.md). TL;DR: Python 3.11 venv (`tools/setup_venv.sh`); torch from the
**cu128** index (RTX 5070 Ti / Blackwell); dataset at `D:\DataSet\LRS3`; Vitis/Vivado 2024.2 for Phase 3.
