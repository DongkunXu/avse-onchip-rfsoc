"""summarize_efficiency.py — cross-platform measured data table + figure for the C7 AVSE.

Reads the per-platform JSONs in experiments/power_efficiency/ (bench_cuda_b1, bench_cuda_b64, bench_cpu_b1
from bench_inference.py; bench_fpga from the board's bench_fpga_power.py) and prints/plots the MEASURED
numbers only: latency, throughput, real-time factor, power, energy-per-window, energy-per-audio-second,
perf-per-watt. No estimates, no derived scenarios. Power: FPGA = whole board (INA220), GPU = card (NVML),
CPU = not measured.
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
    args = ap.parse_args()
    d = Path(args.dir)

    def load(name):
        p = d / f"{name}.json"
        return json.load(open(p)) if p.exists() else None

    specs = [("fpga", "FPGA RFSoC 4x2", "int16"),
             ("cuda_b1", "GPU RTX 5070 Ti (b1)", "fp32"),
             ("cuda_b64", "GPU RTX 5070 Ti (b64)", "fp32"),
             ("cpu_b1", "CPU i5-14600KF (b1)", "fp32")]
    rows = []
    for key, label, prec in specs:
        r = load("bench_" + key)
        if not r:
            print(f"NOTE: missing bench_{key}.json (skipped)"); continue
        lat = r["latency_ms_per_window"]; thr = r["throughput_win_per_s"]; rt = r["realtime_factor"]
        pw = r.get("power_w")                                # None for CPU (not measured)
        e_win = (pw * (lat / 1000.0)) if pw is not None else None
        rows.append({"label": label, "prec": prec, "lat_ms": lat, "thr": thr, "rt": rt,
                     "power_w": pw, "e_win_j": e_win,
                     "e_audio_j": (e_win / AUDIO_S) if e_win is not None else None,
                     "perf_per_w": (thr / pw) if pw is not None else None,
                     "idle_w": r.get("idle_power_w"), "active_w": r.get("power_w"),
                     "dyn_w": r.get("dynamic_power_w"), "per_rail_w": r.get("per_rail_active_w"),
                     "is_fpga": key == "fpga"})

    def cell(v, fmt):
        return (fmt.format(v)) if v is not None else "—"

    print("\n===== MEASURED (FPGA power = whole board / INA220; GPU power = card / NVML; CPU power = not measured) =====")
    h = (f"{'platform':22s}{'prec':6s}{'lat/win':>10s}{'win/s':>9s}{'xRT':>8s}"
         f"{'power_W':>9s}{'J/win':>8s}{'J/audio-s':>11s}{'win/s/W':>9s}")
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['label']:22s}{r['prec']:6s}{r['lat_ms']:>9.2f}m{r['thr']:>9.1f}{r['rt']:>8.1f}"
              f"{cell(r['power_w'], '{:.1f}'):>9s}{cell(r['e_win_j'], '{:.2f}'):>8s}"
              f"{cell(r['e_audio_j'], '{:.3f}'):>11s}{cell(r['perf_per_w'], '{:.2f}'):>9s}")
    fpga = next((r for r in rows if r["is_fpga"]), None)
    if fpga:
        print(f"\nFPGA power rails: idle {fpga['idle_w']:.2f} W, active {fpga['active_w']:.2f} W, "
              f"dynamic +{fpga['dyn_w']:.2f} W.")
        if fpga["per_rail_w"]:
            print("  per-rail active (W): " + ", ".join(f"{k}={v:.2f}" for k, v in fpga["per_rail_w"].items()))

    # ---- figure: measured-power platforms only ----
    mp = [r for r in rows if r["power_w"] is not None]
    labels = [r["label"].replace(" RTX 5070 Ti", "").replace(" i5-14600KF", "").replace("RFSoC 4x2", "")
              for r in mp]
    colors = ["#1b7837" if r["is_fpga"] else "#3b6fb6" for r in mp]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for a, (key, title, fmt) in zip(ax, [("power_w", "Power (W)", "{:.1f}"),
                                         ("lat_ms", "Latency / window (ms)", "{:.1f}"),
                                         ("e_win_j", "Energy / window (J)", "{:.2f}")]):
        vals = [r[key] for r in mp]
        bars = a.bar(labels, vals, color=colors)
        a.set_yscale("log"); a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.3)
        a.tick_params(axis="x", labelsize=8, rotation=15)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    fig.suptitle("C7 AVSE inference — measured (FPGA int16 / GPU fp32), one 1.2 s window", fontsize=10.5)
    fig.tight_layout()
    out = d / "efficiency.png"; fig.savefig(out, dpi=140)
    json.dump({"audio_s_per_window": AUDIO_S, "rows": rows}, open(d / "efficiency_summary.json", "w"), indent=2)
    print(f"\nsaved {out} + efficiency_summary.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
