"""plot_snr_bins.py — plot the per-SNR-bin on-board eval (SI-SDR / PESQ / STOI vs input SNR),
FPGA vs int16 software vs mixed input. Reads snr_bin_results.json from score_board_snr_bins.py.

  python tools/plot_snr_bins.py --dir hw/board/snr_eval
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw/board/snr_eval")
    args = ap.parse_args()
    d = Path(args.dir)
    r = json.load(open(d / "snr_bin_results.json"))
    lo = np.array([float(x) for x in r["fpga"]["per_bin"]["0"]["label"].split("[")[1].split("_to_")[0:1]])  # noqa
    # bin midpoints from labels
    mids = []
    for b in range(len(r["fpga"]["per_bin"])):
        lab = r["fpga"]["per_bin"][str(b)]["label"]
        a, c = lab.split("[")[1].rstrip("]").split("_to_")
        mids.append((float(a) + float(c)) / 2)
    mids = np.array(mids)
    has_emu = "emulator" in r

    def col(tag, m):
        pb = r[tag]["per_bin"]
        return [pb[str(b)][m] for b in range(len(pb))]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (me, mm, name) in enumerate([("si_e", "si_m", "SI-SDR (dB)"),
                                        ("p_e", "p_m", "PESQ-WB"), ("s_e", "s_m", "STOI")]):
        ax[j].plot(mids, col("fpga", me), "o-", label="FPGA (enhanced)", color="C0")
        if has_emu:
            ax[j].plot(mids, col("emulator", me), "s--", label="int16 sw", color="C1", alpha=.8)
        ax[j].plot(mids, col("fpga", mm), "^:", label="mixed (input)", color="0.5")
        ax[j].set_xlabel("input SNR (dB)"); ax[j].set_title(name); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("On-board AVSE per SNR bin (optimized bitstream)", fontsize=11)
    fig.tight_layout()
    out = d / "snr_trend_onboard.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
