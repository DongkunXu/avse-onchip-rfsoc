"""summarize_efficiency.py — cross-platform power/efficiency table + figure for the C7 AVSE.

Reads the per-platform JSONs in experiments/power_efficiency/ (bench_cuda_b1, bench_cuda_b64, bench_cpu_b1
from bench_inference.py; bench_fpga from the board's bench_fpga_power.py) and produces:
  - a printed comparison table (latency / throughput / real-time factor / power / energy-per-window /
    energy-per-audio-second / perf-per-watt),
  - a device-level + a system-level (host-included) view, and
  - efficiency.png (power, latency, energy bars, log scale).

Power scopes (stated honestly): FPGA = whole RFSoC board incl. PS+DDR (INA220 measured); GPU = card only
(NVML measured); CPU = package (ESTIMATED — Windows exposes no CPU power telemetry here). The CPU power and
the host-power for the system-level view are explicit assumptions, flagged as such.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
AUDIO_S = 19200 / 16000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "experiments/power_efficiency"))
    ap.add_argument("--cpu-power-w", type=float, default=70.0, help="ESTIMATE for CPU package power (W)")
    ap.add_argument("--host-idle-w", type=float, default=70.0, help="ESTIMATE host idle power for GPU/CPU system view (W)")
    args = ap.parse_args()
    d = Path(args.dir)

    def load(name):
        p = d / f"{name}.json"
        return json.load(open(p)) if p.exists() else None

    rows = []
    specs = [("fpga", "FPGA RFSoC 4x2", "int16", False),
             ("cuda_b1", "GPU RTX 5070 Ti (b1)", "fp32", False),
             ("cuda_b64", "GPU RTX 5070 Ti (b64)", "fp32", False),
             ("cpu_b1", "CPU i5-14600KF (b1)", "fp32", True)]
    for key, label, prec, is_cpu in specs:
        r = load("bench_" + key)
        if not r:
            print(f"NOTE: missing bench_{key}.json (skipped)"); continue
        lat = r["latency_ms_per_window"]; thr = r["throughput_win_per_s"]; rt = r["realtime_factor"]
        if r.get("power_w") is not None:
            pw = r["power_w"]; pw_measured = True
        else:
            pw = args.cpu_power_w; pw_measured = False     # CPU estimate
        e_win = pw * (lat / 1000.0)                          # J/window (device-level)
        rows.append({"label": label, "prec": prec, "lat_ms": lat, "thr": thr, "rt": rt,
                     "power_w": pw, "power_measured": pw_measured, "e_win_j": e_win,
                     "e_audio_j": e_win / AUDIO_S, "perf_per_w": thr / pw,
                     "idle_w": r.get("idle_power_w"), "dyn_w": r.get("dynamic_power_w"),
                     "is_cpu": is_cpu, "is_fpga": key == "fpga"})

    # ---- device-level table ----
    print("\n================= DEVICE-LEVEL (power scope differs — see notes) =================")
    h = f"{'platform':22s}{'prec':6s}{'lat/win':>10s}{'win/s':>9s}{'xRT':>8s}{'power_W':>9s}{'J/win':>8s}{'J/audio-s':>11s}{'win/s/W':>9s}"
    print(h); print("-" * len(h))
    for r in rows:
        pw = f"{r['power_w']:.1f}" + ("" if r["power_measured"] else "*")
        print(f"{r['label']:22s}{r['prec']:6s}{r['lat_ms']:>9.2f}m{r['thr']:>9.1f}{r['rt']:>8.1f}{pw:>9s}"
              f"{r['e_win_j']:>8.2f}{r['e_audio_j']:>11.3f}{r['perf_per_w']:>9.2f}")
    print("* CPU power is an ESTIMATE (no telemetry); all others measured (FPGA=INA220 board, GPU=NVML card).")

    fpga = next((r for r in rows if r["is_fpga"]), None)
    if fpga:
        print(f"\nFPGA detail: idle {fpga['idle_w']:.2f} W, active {fpga['power_w']:.2f} W "
              f"(dynamic +{fpga['dyn_w']:.2f} W) → dynamic energy {fpga['dyn_w']*fpga['lat_ms']/1000*1000:.0f} mJ/win.")

    # ---- system-level (host included for GPU/CPU) ----
    print("\n================= SYSTEM-LEVEL (GPU/CPU need a host ≈ +{:.0f} W; FPGA is standalone) =========".format(args.host_idle_w))
    for r in rows:
        sys_w = r["power_w"] if r["is_fpga"] else r["power_w"] + args.host_idle_w
        sys_e = sys_w * (r["lat_ms"] / 1000.0)
        print(f"  {r['label']:22s} system ≈ {sys_w:6.1f} W → {sys_e:5.2f} J/window"
              + ("  (standalone, measured)" if r["is_fpga"] else "  (+host est.)"))

    # ---- always-on SINGLE real-time stream (the actual edge deployment) ----
    # To serve 1 live AV stream you process 1.2 s of audio every 1.2 s: busy fraction = latency/AUDIO_S,
    # idle the rest. The GPU/CPU + host cannot power off between windows, so their idle power dominates and
    # their speed is wasted (you only ever need 1x real-time). Energy per second-of-audio (= per s wall):
    GPU_IDLE_W = 13.6   # measured nvidia-smi idle
    print("\n========== ALWAYS-ON, ONE REAL-TIME STREAM (the deployment) — energy per second of audio ==========")
    print("  (busy fraction = lat/1.2 s; rest idle; GPU/CPU include +{:.0f} W host idle, FPGA standalone)".format(args.host_idle_w))
    ao = []
    for r in rows:
        duty = (r["lat_ms"] / 1000.0) / AUDIO_S
        if r["is_fpga"]:
            p_act, p_idle = r["power_w"], (r["idle_w"] or r["power_w"])
        elif r["is_cpu"]:
            p_act, p_idle = r["power_w"] + args.host_idle_w, args.host_idle_w
        else:
            p_act, p_idle = r["power_w"] + args.host_idle_w, GPU_IDLE_W + args.host_idle_w
        e_audio_s = p_act * duty + p_idle * (1 - duty)
        ao.append(e_audio_s)
        print(f"  {r['label']:22s} busy {duty*100:5.1f}% → {e_audio_s:6.1f} J per audio-second"
              + ("  (measured)" if r["is_fpga"] else "  (+host est.)"))
    fpga_ao = ao[0]
    for r, e in zip(rows, ao):
        if not r["is_fpga"]:
            print(f"     → FPGA is {e/fpga_ao:.1f}x more energy-efficient than {r['label']} for one live stream")

    # ---- figure ----
    labels = [r["label"].replace(" RTX 5070 Ti", "").replace(" i5-14600KF", "").replace("RFSoC 4x2", "")
              for r in rows]
    colors = ["#1b7837" if r["is_fpga"] else ("#999999" if r["is_cpu"] else "#3b6fb6") for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, (key, title, fmt) in zip(ax, [("power_w", "Power (W) — lower better", "{:.1f}"),
                                         ("lat_ms", "Latency / window (ms) — lower better", "{:.1f}"),
                                         ("e_win_j", "Energy / window (J) — lower better", "{:.2f}")]):
        vals = [r[key] for r in rows]
        bars = a.bar(labels, vals, color=colors)
        a.set_yscale("log"); a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.3)
        a.tick_params(axis="x", labelsize=8, rotation=20)
        for b, v, r in zip(bars, vals, rows):
            star = "*" if (key == "power_w" and not r["power_measured"]) else ""
            a.text(b.get_x() + b.get_width()/2, v, fmt.format(v) + star, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("C7 AVSE inference efficiency: FPGA (int16, measured) vs GPU/CPU (fp32) — one 1.2 s window  "
                 "(green=FPGA; *=CPU power estimate)", fontsize=10.5)
    fig.tight_layout()
    out = d / "efficiency.png"; fig.savefig(out, dpi=140)
    json.dump({"rows": rows, "audio_s": AUDIO_S, "cpu_power_est_w": args.cpu_power_w,
               "host_idle_est_w": args.host_idle_w}, open(d / "efficiency_summary.json", "w"), indent=2)
    print(f"\nsaved {out} + efficiency_summary.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
