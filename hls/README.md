# hls/ — Phase 3: HLS C++ (finalists only)

**Placeholder.** Populated only after Phase 2 produces a Pareto-frontier winner and the owner picks
1–3 finalists for hardware validation. See [`../docs/ROADMAP.md`](../docs/ROADMAP.md) Phase 3.

What will live here: the Vitis HLS C++ implementation of the chosen architecture (int16 `ap_fixed`),
its testbench, and the per-IP csynth/export `.tcl`. Toolchain: Vitis HLS 2024.2.

Inherit the validated techniques (do not reinvent) from
[`../docs/reference/prior-wins.md`](../docs/reference/prior-wins.md): the shared dwsep-conv engine,
math-exact buffer elimination, 2D-weight-flatten, URAM packing / partition-factor BRAM reclaim.

**Do not start writing HLS until a finalist exists and the owner has approved Phase 3.**
