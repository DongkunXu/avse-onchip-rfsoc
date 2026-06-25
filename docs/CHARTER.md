# CHARTER — what this project is and is not

**Created**: 2026-06-25
**Status**: pre-Phase-1 (scaffolding complete; no research code yet)

This is the single source of truth for *why this project exists* and *what counts as success*.
It is distilled from the reference project's scoping note (`../UNet-AVSE-Vitis/NEW_PROJECT_ONCHIP_AVSE.md`)
and re-stated here so this repository is self-contained.

---

## 1. The goal

Design, from scratch, an audio-visual speech enhancement (AVSE) system that **fits entirely on the
programmable logic of a single mid-size FPGA (Real Digital RFSoC 4x2 / Xilinx ZU48DR) in ONE static
configuration** — no partial reconfiguration, no swapping bitstreams, no time-multiplexing the chip
across phases.

The reference deployment fails this: chopped into **4 separate full bitstreams** loaded sequentially
via PCAP because all four together need **~215 % of the chip's BRAM**. This project must close that
>2× gap so the whole pipeline **co-resides**.

## 2. Priority order (hard, from the project owner — do not reorder)

1. **FIT IS THE HARD CONSTRAINT.** The entire system must fit on-chip, single-config. If it does not
   fit, nothing else matters — quality is moot.
2. **The contribution is the HARDWARE ARCHITECTURE.** Publication target is a **circuits / hardware
   journal**. Novelty in the *dataflow, memory hierarchy, data-movement paradigm, and data-structure
   choices* that make a full AVSE fit is worth far more than enhancement quality.
   - Weak (machine learning, not circuits): *"we changed the ML model so it's smaller."*
   - Strong (circuits): *"we designed a novel on-chip dataflow/memory architecture that lets a
     complete AVSE pipeline run in a single FPGA configuration at X % resources, vs a 4-way-reconfig
     baseline."*
   - The model architecture is a **co-design knob in service of the hardware story**, not the center.
3. **Quality is secondary but maximized within the fit constraint.** Sacrificing speech-enhancement
   quality is explicitly allowed. Get it as good as possible *given* that the whole thing fits — but
   **never trade "fits" for "sounds better."**

## 3. The binding objective (the one number we minimize)

> **Peak simultaneously-live activation working set** = Σ (C × T) over co-resident tensors.

This is *analytically computable* for any candidate architecture before building it (see
[`../analysis`](../analysis)). The whole project is, in effect, a search for an architecture +
dataflow that drives this number below the on-chip memory budget while preserving as much quality as
possible.

## 4. What this is NOT

- **Not a continuation** of the reference U-Net deployment. Do not extend it, refactor it, or feel
  bound by its U-Net structure, its int16 HLS code, or its 4-bitstream scheme.
- **Not a weight-compression project.** The wall is activations, not weights (see
  [`reference/bottleneck-diagnosis.md`](reference/bottleneck-diagnosis.md)). Pruning/quantizing
  *weights* does not solve it.
- **Not a precision-reduction project.** int16 is locked; int8/DPU was ruled out
  ([`reference/dead-ends.md`](reference/dead-ends.md)).
- **Not a board-change project.** Single-board (RFSoC 4x2 / ZU48DR) is assumed fixed. AIE/Versal is a
  different project and only revisited if the owner changes hardware.

## 5. Mental-frame discipline

The point of starting fresh is to **escape the structure that created the problem.** Do not reason
"how do I shrink the existing U-Net" — that reproduces its constraints. Reason instead: *"what
architecture + dataflow makes a whole AVSE system fit on-chip in one static configuration?"* The
existing model is **one data point, not the design center.**

## 6. Open decisions reserved for the owner

These are **not** to be settled unilaterally (tracked in [`DECISIONS.md`](DECISIONS.md)):

1. **Quality floor** — *answered*: fit is hard; quality is "as good as possible given it fits," no
   fixed dB floor. The Phase-2 Pareto frontier will expose the trade and the owner picks a point.
2. **Time-domain vs STFT / frequency-domain** — *open*. The most fundamental (root-cause) change but
   the biggest departure from the reference. **Needs an explicit owner call before Phase 2.**
3. **Single-board constraint** — *assumed fixed* (RFSoC 4x2 / ZU48DR).
