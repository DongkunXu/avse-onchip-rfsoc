# Power / energy / efficiency — FPGA vs GPU vs CPU (C7 AVSE inference)

An independent measurement of **average inference power, energy, and efficiency** of the deployed AVSE
running on the **RFSoC 4x2 FPGA** (the optimized single-static-config bitstream, int16) versus the same
model on this workstation's **GPU (RTX 5070 Ti)** and **CPU (i5-14600KF)** in FP32. One "inference" =
one **1.2 s window** (19 200 samples @ 16 kHz).

> TL;DR — **honest verdict.** For *raw throughput and energy-per-inference run flat-out*, the GPU wins
> decisively (the model is tiny → it underutilizes a 300 W GPU but is still ~100–460× faster). The FPGA's
> wins are: (1) **lowest power by far — 6.6 W measured** (vs 76–209 W GPU); (2) for the **actual deployment
> — one always-on real-time stream — it is ~12–14× more energy-efficient**, because a GPU/CPU + host idle at
> ~70–84 W while their speed is wasted on a single stream; (3) it is a **complete standalone chip**, no host.
> The project's contribution was always *fit one chip*, not *beat a GPU on throughput* — and at the system
> level for an edge AVSE node, low power is exactly where it wins.

## How measured (honest about scope)

| platform | precision | power source | scope |
|---|---|---|---|
| **FPGA** RFSoC 4x2 | int16 (deployed) | **measured** — 9× on-board INA220 rails (`/sys/class/hwmon`), idle vs active during a 480-inference loop on real data | whole board (PL **+ PS + DDR**) |
| **GPU** RTX 5070 Ti | fp32 | **measured** — NVML `power.draw` sampled during a warmed-up loop | GPU **card only** (excludes host) |
| **CPU** i5-14600KF | fp32 | **ESTIMATE** (Windows exposes no CPU power telemetry here) — package ~70 W under load | CPU package (excludes host GPU) |

Tools: `tools/bench_inference.py` (GPU/CPU), `hw/board/bench_fpga_power.py` (board, INA220),
`tools/summarize_efficiency.py` (table + figure). Raw per-platform JSON + `efficiency.png` in this folder.
FP32 on GPU/CPU vs int16 on FPGA = each platform's **as-deployed** path (a GPU would use FP32/TensorRT, not
int16). Power scopes differ — read the verdict, not a single cell.

## Device-level results (measured except CPU power*)

| platform | prec | latency/win | throughput | ×real-time | power | energy/win | energy/audio-s | perf/W (win/s/W) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **FPGA RFSoC 4x2** | int16 | **286 ms** | 3.5 win/s | **4.2×** | **6.6 W** | 1.90 J | 1.58 J | 0.53 |
| GPU 5070 Ti (b=1) | fp32 | 2.94 ms | 340 win/s | 408× | 75.8 W | 0.22 J | 0.19 J | 4.49 |
| GPU 5070 Ti (b=64) | fp32 | **0.62 ms** | **1617 win/s** | **1941×** | 209 W | **0.13 J** | **0.11 J** | **7.72** |
| CPU i5-14600KF (b=1) | fp32 | 17.1 ms | 58 win/s | 70× | ~70 W* | ~1.20 J* | ~1.00 J* | 0.84 |

\* CPU power is an estimate. FPGA detail: **idle 5.90 W → active 6.63 W (dynamic only +0.73 W)**; the PL
compute adds <1 W — most of the 6.6 W is the always-on PS/Linux/DDR. (The Vivado vectorless estimate was
11.6 W — pessimistic vs the 6.6 W measured.)

**Reading it:** run flat-out, the FPGA has the **highest** per-window energy (1.90 J) simply because it's
slowest; the GPU's 100–460× speed makes its per-inference energy lower even at 12–32× the power. So on the
throughput/energy axes **the FPGA loses** — expected for a 308 k-param model on a 300 W GPU, and the FPGA was
built for *fit*, not peak speed (II not fully 1, 187.5 MHz).

## The deployment that matters: one always-on real-time stream

An edge AVSE node serves **one** live audio-visual stream — it needs **1× real-time**, not 1600 win/s. Then
the GPU/CPU + host **cannot power down between windows** and idle at ~70–84 W while doing ~0.1–1.4 % useful
work; the FPGA is a self-contained 6.6 W node. Energy **per second of audio** (busy = latency/1.2 s, rest idle;
GPU/CPU + ~70 W host idle; FPGA standalone measured):

| platform | busy duty | **energy / audio-second** | vs FPGA |
|---|--:|--:|--:|
| **FPGA RFSoC 4x2** | 23.8 % | **6.1 J** | — |
| GPU 5070 Ti + host | 0.2 % | 83.7 J | **13.8× worse** |
| CPU i5-14600KF system | 1.4 % | 71.0 J | **11.7× worse** |

**For the real use case the FPGA is ~12–14× more energy-efficient.** (Host-idle 70 W and CPU power are
estimates; the FPGA and GPU-card numbers are measured. The point is robust to the exact host figure: the
GPU's speed is simply unusable for a single stream, so its idle power sets the cost.)

## Bottom line
- **Throughput / energy-per-inference, flat-out:** GPU wins (use a GPU if you batch many streams). FPGA loses.
- **Absolute power:** FPGA wins — **6.6 W** complete system vs 76–209 W GPU card (+ its host).
- **Energy for one always-on real-time stream (the deployment):** **FPGA wins ~12–14×.**
- **Form factor:** FPGA is one self-contained chip (no host); deterministic 286 ms latency; single static config.

## Reproduce
```
python tools/bench_inference.py --device cuda --batch 1 --iters 2500 --warmup 200
python tools/bench_inference.py --device cuda --batch 64 --iters 100
python tools/bench_inference.py --device cpu  --batch 1  --iters 60
# board: scp hw/board/bench_fpga_power.py, then (sudo) python3 bench_fpga_power.py --loops 30
python tools/summarize_efficiency.py
```
