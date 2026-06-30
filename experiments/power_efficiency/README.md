# Power / energy / efficiency — measured data (C7 AVSE inference)

Inference of the deployed AVSE model. One inference = one **1.2 s window** (19 200 samples @ 16 kHz).
Platforms: **FPGA RFSoC 4x2** (int16, optimized bitstream), **GPU RTX 5070 Ti** (fp32), **CPU i5-14600KF**
(fp32). Numbers below are measured (CPU power is not measured — no telemetry available here).

Power measurement scope: FPGA = whole board (PL + PS + DDR), 9× on-board INA220 rails
(`/sys/class/hwmon/*/power1_input`); GPU = card only, NVML `power.draw`; CPU = not measured.
energy/window = power × latency; energy/audio-s = energy/window ÷ 1.2; perf/W = throughput ÷ power.

| platform | precision | batch | latency/win | throughput | ×real-time | power | energy/win | energy/audio-s | perf/W (win/s/W) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| FPGA RFSoC 4x2 | int16 | 1 | 286.2 ms | 3.49 win/s | 4.2× | 6.63 W | 1.90 J | 1.581 J | 0.53 |
| GPU RTX 5070 Ti | fp32 | 1 | 2.94 ms | 340.4 win/s | 408.5× | 75.8 W | 0.22 J | 0.186 J | 4.49 |
| GPU RTX 5070 Ti | fp32 | 64 | 0.62 ms | 1617.3 win/s | 1940.8× | 209.4 W | 0.13 J | 0.108 J | 7.72 |
| CPU i5-14600KF | fp32 | 1 | 17.08 ms | 58.5 win/s | 70.2× | — | — | — | — |

FPGA board power (INA220): idle 5.90 W, active 6.63 W, dynamic +0.73 W. Per-rail active (W):
hwmon0 4.00, hwmon1 1.40, hwmon2 1.07, hwmon3 0.00, hwmon4 0.04, hwmon5 0.06, hwmon6 0.00, hwmon7 0.04,
hwmon8 0.02. Vivado vectorless estimate (`hw/reports/postroute_power.rpt`): 11.6 W on-chip.

Raw per-platform JSON: `bench_fpga.json`, `bench_cuda_b1.json`, `bench_cuda_b64.json`, `bench_cpu_b1.json`.
Figure: `efficiency.png`. Aggregated: `efficiency_summary.json`.

## Reproduce
```
python tools/bench_inference.py --device cuda --batch 1 --iters 2500 --warmup 200
python tools/bench_inference.py --device cuda --batch 64 --iters 100
python tools/bench_inference.py --device cpu  --batch 1  --iters 60
# board: scp hw/board/bench_fpga_power.py, then (sudo) python3 bench_fpga_power.py --loops 30
python tools/summarize_efficiency.py
```
