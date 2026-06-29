"""plot_video_ablation.py — per-SNR-bin visual-ablation charts (real video vs video zeroed).

Three modes:
  --realm fp32  : FP32/Python ablation  (video_ablation_results.json: fp32_video / fp32_novideo)
  --realm fpga  : on-board FPGA ablation (board_video_ablation_results.json: fpga_video / fpga_novideo)
  --realm both  : combined 4-curve overlay (FP32 + FPGA, each with/without video) — shows the visual
                  contribution is preserved on real silicon.

Each panel (SI-SDR / PESQ / STOI vs input SNR) draws the with-video (solid) vs video-zeroed (dashed)
curves with a scene-count-weighted-average line; the shaded band is the visual contribution. Honest:
only series present in the JSON(s) are plotted.

  python tools/plot_video_ablation.py --dir hw/board/snr_eval --realm fp32
  python tools/plot_video_ablation.py --dir hw/board/snr_eval --realm fpga
  python tools/plot_video_ablation.py --dir hw/board/snr_eval --realm both
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("si_e", "SI-SDR (dB)"), ("p_e", "PESQ-WB"), ("s_e", "STOI")]

# realm -> (results filename, [(key, label, color, ls, marker), ...], out filename, title prefix)
REALMS = {
    "fp32": ("video_ablation_results.json",
             [("fp32_video", "FP32: with video", "C0", "-", "o"),
              ("fp32_novideo", "FP32: video zeroed", "C3", "--", "x")],
             "video_ablation.png", "FP32 (Python)"),
    "fpga": ("board_video_ablation_results.json",
             [("fpga_video", "FPGA: with video", "C0", "-", "D"),
              ("fpga_novideo", "FPGA: video zeroed", "C3", "--", "x")],
             "board_video_ablation.png", "on-board FPGA"),
}


def mids_of(rj_series):
    nb = len(rj_series["per_bin"])
    out = []
    for b in range(nb):
        a, c = rj_series["per_bin"][str(b)]["label"].split("[")[1].rstrip("]").split("_to_")
        out.append((float(a) + float(c)) / 2)
    return np.array(out)


def col(series, m):
    pb = series["per_bin"]; nb = len(pb)
    return np.array([pb[str(b)][m] if pb[str(b)][m] is not None else np.nan for b in range(nb)])


def draw(ax_row, series_list, mids, shade=True):
    """series_list: [(json_obj_for_series, label, color, ls, marker)]"""
    for j, (m, name) in enumerate(METRICS):
        a = ax_row[j]
        ys = []
        for sj, label, color, ls, mk in series_list:
            y = col(sj, m); ys.append(y)
            a.plot(mids, y, marker=mk, ls=ls, color=color, ms=5, lw=1.9, label=label)
            avg = sj["overall"]["weighted"][m]
            if avg is not None:
                a.axhline(avg, color=color, ls=":", lw=1.0, alpha=0.6)
        if shade and len(ys) >= 2:
            a.fill_between(mids, ys[1], ys[0], where=~(np.isnan(ys[0]) | np.isnan(ys[1])),
                           color="C0", alpha=0.10, interpolate=True)
        a.set_xlabel("input SNR (dB)"); a.set_title(name); a.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw/board/snr_eval")
    ap.add_argument("--realm", choices=["fp32", "fpga", "both"], default="fpga")
    args = ap.parse_args()
    d = Path(args.dir)

    if args.realm in ("fp32", "fpga"):
        fn, spec, outfn, title = REALMS[args.realm]
        r = json.load(open(d / fn))
        present = [(r[k], lab, c, ls, mk) for (k, lab, c, ls, mk) in spec if k in r]
        if len(present) < 2:
            print(f"ERROR: need both arms in {fn} (have {[s[1] for s in present]})"); return 1
        mids = mids_of(present[0][0])
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        draw(ax, present, mids)
        ax[0].legend(fontsize=8.5, loc="upper left")
        wv = present[0][0]["overall"]["weighted"]; wn = present[1][0]["overall"]["weighted"]
        fig.suptitle(f"Visual ablation — {title} (665-scene SNR-bin set): real video vs video zeroed — "
                     f"Δweighted SI-SDR {wn['si_e']-wv['si_e']:+.2f} dB, PESQ {wn['p_e']-wv['p_e']:+.3f}, "
                     f"STOI {wn['s_e']-wv['s_e']:+.3f}  (shaded = visual contribution)", fontsize=10.5)
    else:  # both
        rf = json.load(open(d / REALMS["fp32"][0]))
        rb = json.load(open(d / REALMS["fpga"][0]))
        present = []
        for k, lab, c, ls, mk in REALMS["fp32"][1]:
            if k in rf: present.append((rf[k], lab, c, ls, mk))
        for k, lab, c, ls, mk in REALMS["fpga"][1]:
            if k in rb: present.append((rb[k], lab, c, ls, mk))
        # recolor for 4-curve clarity: FP32 with/without = C0/C3 solid/dash; FPGA = C2/C1
        recolor = {"FP32: with video": ("C0", "-", "o"), "FP32: video zeroed": ("C0", "--", "o"),
                   "FPGA: with video": ("C2", "-", "D"), "FPGA: video zeroed": ("C2", "--", "D")}
        present = [(sj, lab, *recolor.get(lab, (c, ls, mk))) for (sj, lab, c, ls, mk) in present]
        if not present:
            print("ERROR: no series for combined plot"); return 1
        mids = mids_of(present[0][0])
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        draw(ax, present, mids, shade=False)
        ax[0].legend(fontsize=8, loc="upper left")
        fig.suptitle("Visual ablation, FP32 vs on-board FPGA (with video solid, video zeroed dashed) — "
                     "the video contribution holds on real silicon", fontsize=10.5)
        outfn = "video_ablation_combined.png"

    fig.tight_layout()
    out = d / outfn
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
