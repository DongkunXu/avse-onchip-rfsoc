"""plot_video_ablation.py — per-SNR-bin visual-ablation chart (FP32, pure Python).

Two series per panel: the normal model (real video, solid) vs video zeroed (dashed). 3 panels:
SI-SDR / PESQ / STOI vs input SNR. Each series gets a horizontal scene-count-weighted-average line.
The shaded gap between the curves IS the visual contribution. Honest: only series present in the
JSON are plotted.

Reads video_ablation_results.json (from eval_video_ablation_snr_bins.py).

  python tools/plot_video_ablation.py --dir hw/board/snr_eval
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (json key, legend label, color, linestyle, marker)
SERIES = [
    ("fp32_video",   "with video (FP32)",   "C0", "-",  "o"),
    ("fp32_novideo", "video zeroed (FP32)", "C3", "--", "x"),
]
METRICS = [("si_e", "SI-SDR (dB)"), ("p_e", "PESQ-WB"), ("s_e", "STOI")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw/board/snr_eval")
    args = ap.parse_args()
    d = Path(args.dir)
    r = json.load(open(d / "video_ablation_results.json"))

    present = [s for s in SERIES if s[0] in r]
    missing = [s[0] for s in SERIES if s[0] not in r]
    if missing:
        print(f"NOTE: series missing (NOT fabricated, omitted): {missing}")
    if len(present) < 2:
        print("ERROR: need both arms to show an ablation"); return 1
    ref = present[0][0]
    nb = len(r[ref]["per_bin"])
    mids = []
    for b in range(nb):
        a, c = r[ref]["per_bin"][str(b)]["label"].split("[")[1].rstrip("]").split("_to_")
        mids.append((float(a) + float(c)) / 2)
    mids = np.array(mids)

    def col(tag, m):
        pb = r[tag]["per_bin"]
        return np.array([pb[str(b)][m] if pb[str(b)][m] is not None else np.nan for b in range(nb)])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for j, (m, name) in enumerate(METRICS):
        a = ax[j]
        ys = {}
        for tag, label, color, ls, mk in present:
            y = col(tag, m)
            ys[tag] = y
            a.plot(mids, y, marker=mk, ls=ls, color=color, ms=5, lw=1.9, label=label)
            avg = r[tag]["overall"]["weighted"][m]
            if avg is not None:
                a.axhline(avg, color=color, ls=":", lw=1.1, alpha=0.7)
                a.annotate(f"{avg:.2f}" if m == "si_e" else f"{avg:.3f}",
                           xy=(mids[0 if tag == present[0][0] else -1], avg), fontsize=7.5,
                           color=color, fontweight="bold", ha="left" if tag == present[0][0] else "right",
                           va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        # shade the visual-contribution gap (with-video above no-video)
        yv, yn = ys[present[0][0]], ys[present[1][0]]
        a.fill_between(mids, yn, yv, where=~(np.isnan(yv) | np.isnan(yn)),
                       color="C0", alpha=0.10, interpolate=True)
        a.set_xlabel("input SNR (dB)"); a.set_title(name); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8.5, loc="upper left")
    wv = r["fp32_video"]["overall"]["weighted"]; wn = r["fp32_novideo"]["overall"]["weighted"]
    fig.suptitle("Visual ablation (FP32, 665-scene SNR-bin set): real video vs video zeroed — "
                 f"Δweighted SI-SDR {wn['si_e']-wv['si_e']:+.2f} dB, PESQ {wn['p_e']-wv['p_e']:+.3f}, "
                 f"STOI {wn['s_e']-wv['s_e']:+.3f}  (shaded = visual contribution)", fontsize=10.5)
    fig.tight_layout()
    out = d / "video_ablation.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
