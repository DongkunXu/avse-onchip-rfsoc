"""score_board_snr_bins.py — PC-side: score the FPGA's per-SNR-bin outputs.

Reads meta.npz + the per-chunk *_outputs.npz (FPGA enhanced) + *_windows.npz (mixed input), computes
per-window SI-SDR / PESQ-WB / STOI (enhanced and the mixed baseline) with the SAME metric functions as
eval_full_dev.py, averages per scene, then per SNR bin, and a scene-count-WEIGHTED overall (weights =
full dev bin scene counts). Optionally also runs the int16 emulator on the same windows (--emulator) for a
software per-bin baseline (FPGA-vs-software comparison).

  python tools/score_board_snr_bins.py --dir hw/board/snr_eval
  python tools/score_board_snr_bins.py --dir hw/board/snr_eval --emulator
"""
import argparse, os, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from avse.metrics import si_sdr, pesq_wb, stoi_score


def metric_one(a):
    """(key, enhanced, mixed, target) -> per-window metrics. Pool worker (no torch import up top)."""
    key, e, m, t = a
    return key, (si_sdr(e, t), si_sdr(m, t), pesq_wb(t, e), pesq_wb(t, m), stoi_score(t, e), stoi_score(t, m))


def aggregate(per_win, sid_arr, bin_arr, labels, full_counts, tag):
    """per_win: {win_idx: 6-tuple}; -> per-scene mean -> per-bin mean -> weighted overall."""
    KEYS = ("si_e", "si_m", "p_e", "p_m", "s_e", "s_m")
    scene = defaultdict(lambda: {k: [] for k in KEYS})
    scene_bin = {}
    for wi, vals in per_win.items():
        sid = sid_arr[wi]; scene_bin[sid] = int(bin_arr[wi])
        for k, v in zip(KEYS, vals):
            if v is not None and np.isfinite(v):
                scene[sid][k].append(v)
    scene_mean = {sid: {k: (float(np.mean(a[k])) if a[k] else None) for k in KEYS} for sid, a in scene.items()}
    nb = len(labels)
    bins = {b: {k: [] for k in KEYS} for b in range(nb)}
    for sid, m in scene_mean.items():
        b = scene_bin[sid]
        for k in KEYS:
            if m[k] is not None:
                bins[b][k].append(m[k])
    per_bin = {}
    for b in range(nb):
        nsc = len(bins[b]["si_e"])
        per_bin[b] = {"label": str(labels[b]), "n_scenes_scored": nsc,
                      **{k: (float(np.mean(bins[b][k])) if bins[b][k] else None) for k in KEYS}}
    # scene-count-weighted overall (weights = full dev bin counts), over bins with data
    def weighted(metric):
        num = den = 0.0
        for b in range(nb):
            v = per_bin[b][metric]
            if v is not None:
                w = float(full_counts[b]); num += w * v; den += w
        return (num / den) if den else None
    # simple over all scored scenes
    def simple(metric):
        vals = [m[metric] for m in scene_mean.values() if m[metric] is not None]
        return float(np.mean(vals)) if vals else None
    overall = {"weighted": {k: weighted(k) for k in KEYS}, "simple": {k: simple(k) for k in KEYS},
               "scenes_scored": len(scene_mean)}
    return {"tag": tag, "per_bin": per_bin, "overall": overall}


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "hw/board/snr_eval"))
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    ap.add_argument("--emulator", action="store_true", help="also score the int16 emulator (software baseline)")
    args = ap.parse_args()
    d = Path(args.dir)
    meta = np.load(d / "meta.npz", allow_pickle=True)
    tgt = meta["target"]; sid = meta["scene_id"]; binx = meta["bin_idx"]
    chunk = meta["chunk"]; row = meta["row"]; labels = meta["bin_labels"]; full = meta["full_bin_scene_counts"]
    N = len(sid)
    # gather FPGA enhanced + mixed per window
    enh = np.zeros((N, tgt.shape[1]), dtype=np.float32); mix = np.zeros_like(enh)
    nch = int(meta["nchunks"])
    for c in range(nch):
        o = np.load(d / f"chunk_{c:03d}_outputs.npz")["audio_out"].astype(np.float32) / 32768.0
        w = np.load(d / f"chunk_{c:03d}_windows.npz")["audio_in"].astype(np.float32) / 32768.0
        m = chunk == c
        idx = np.where(m)[0]
        for wi in idx:
            enh[wi] = o[row[wi]]; mix[wi] = w[row[wi]]
    print(f"scoring {N} windows over {nch} chunks, {args.nproc} procs ...", flush=True)
    pool = mp.Pool(args.nproc)
    fpga = {}
    for key, vals in pool.imap_unordered(metric_one, ((i, enh[i], mix[i], tgt[i]) for i in range(N)), chunksize=8):
        fpga[key] = vals
    res_fpga = aggregate(fpga, sid, binx, labels, full, "FPGA")

    out = {"fpga": res_fpga}
    if args.emulator:
        import torch
        from c7_fixedpoint import C7FixedPoint
        emu = C7FixedPoint(REPO / "experiments/p2-c7-full/deploy_weights.npz", device="cpu",
                           precision="int16", mask="hardsigmoid")
        # feed the SAME int16 inputs (dequantized) through the emulator
        est = np.zeros_like(enh)
        MB = 8  # mini-batch the emulator (video conv intermediates are large)
        for c in range(nch):
            ww = np.load(d / f"chunk_{c:03d}_windows.npz")
            ai = ww["audio_in"].astype(np.float32) / 32768.0
            vi = ww["video_in"].astype(np.float32) / 512.0
            e = np.zeros_like(ai)
            for s in range(0, len(ai), MB):
                with torch.no_grad():
                    e[s:s+MB] = emu.forward(torch.tensor(ai[s:s+MB])[:, None, :],
                                            torch.tensor(vi[s:s+MB])).numpy()[:, 0, :]
            idx = np.where(chunk == c)[0]
            for wi in idx:
                est[wi] = e[row[wi]]
        emud = {}
        for key, vals in pool.imap_unordered(metric_one, ((i, est[i], mix[i], tgt[i]) for i in range(N)), chunksize=8):
            emud[key] = vals
        out["emulator"] = aggregate(emud, sid, binx, labels, full, "emulator(int16 sw)")
    pool.close(); pool.join()

    with open(d / "snr_bin_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for tag in (["fpga", "emulator"] if args.emulator else ["fpga"]):
        r = out[tag]
        print(f"\n================ {r['tag']} — per SNR bin ================")
        print(f"{'bin':28s}{'n':>4s}{'SI-SDR':>9s}{'PESQ':>8s}{'STOI':>7s}{'(mix SI)':>10s}")
        for b in sorted(r["per_bin"], key=int):
            pb = r["per_bin"][b]
            if pb["si_e"] is None: continue
            print(f"{pb['label']:28s}{pb['n_scenes_scored']:>4d}{pb['si_e']:>9.2f}{pb['p_e']:>8.3f}{pb['s_e']:>7.3f}{pb['si_m']:>10.2f}")
        w = r["overall"]["weighted"]; s = r["overall"]["simple"]
        print(f"{'WEIGHTED (by bin scenes)':28s}{'':>4s}{w['si_e']:>9.2f}{w['p_e']:>8.3f}{w['s_e']:>7.3f}{w['si_m']:>10.2f}")
        print(f"{'simple mean (all scenes)':28s}{r['overall']['scenes_scored']:>4d}{s['si_e']:>9.2f}{s['p_e']:>8.3f}{s['s_e']:>7.3f}{s['si_m']:>10.2f}")
    print(f"\nsaved -> {d/'snr_bin_results.json'}", flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
