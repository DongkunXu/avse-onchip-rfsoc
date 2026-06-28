"""plot_snr_bins.py — per-SNR-bin line charts (SI-SDR / PESQ / STOI vs input SNR) for the three
realizations of the model: FP32 original, int16 quantization emulation (dashed), and the on-board FPGA.
Each series also gets a horizontal line at its scene-count-weighted average.

Reads snr_bin_results.json (from score_board_snr_bins.py [+ eval_fp32_snr_bins.py]). HONEST about gaps:
only series actually present in the JSON are plotted, and any missing one is reported, never fabricated.

  python tools/plot_snr_bins.py --dir hw/board/snr_eval
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (json key, legend label, color, linestyle, marker)
SERIES = [
    ("fp32",     "FP32 (original)",          "C2", "-",  "o"),
    ("emulator", "int16 quant. emulation",   "C1", "--", "s"),
    ("fpga",     "on-board (FPGA)",          "C0", "-",  "D"),
]
METRICS = [("si_e", "SI-SDR (dB)"), ("p_e", "PESQ-WB"), ("s_e", "STOI")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw/board/snr_eval")
    args = ap.parse_args()
    d = Path(args.dir)
    r = json.load(open(d / "snr_bin_results.json"))

    present = [s for s in SERIES if s[0] in r]
    missing = [s[0] for s in SERIES if s[0] not in r]
    if missing:
        print(f"NOTE: series missing from {d/'snr_bin_results.json'} (NOT fabricated, omitted): {missing}")
    if not present:
        print("ERROR: no plottable series in results JSON"); return 1
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

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    for j, (m, name) in enumerate(METRICS):
        a = ax[j]
        for si, (tag, label, color, ls, mk) in enumerate(present):
            y = col(tag, m)
            a.plot(mids, y, marker=mk, ls=ls, color=color, ms=5, lw=1.8, label=label)
            avg = r[tag]["overall"]["weighted"][m]            # scene-count-weighted average
            if avg is not None:
                a.axhline(avg, color=color, ls=":", lw=1.1, alpha=0.7)
                # stagger the avg labels horizontally (left/mid/right) so close lines don't collide
                xpos = mids[int(si * (len(mids) - 1) / max(1, len(present) - 1))]
                a.annotate(f"{avg:.2f}", xy=(xpos, avg), fontsize=7, color=color, fontweight="bold",
                           ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        a.set_xlabel("input SNR (dB)"); a.set_title(name); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("AVSE per SNR bin — FP32 vs int16 quant. emulation vs on-board FPGA "
                 "(dotted = scene-count-weighted average)", fontsize=11)
    fig.tight_layout()
    out = d / "snr_trend_onboard.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}  (series: {[s[0] for s in present]})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
