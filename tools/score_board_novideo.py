"""score_board_novideo.py — ON-CHIP visual ablation: score the FPGA with-video vs zeroed-video outputs.

A/B on real silicon, mirroring the FP32 ablation (eval_video_ablation_snr_bins.py) but for the bitstream:
  fpga_video   — the existing with-video board outputs (hw/board/snr_eval/chunk_*_outputs.npz)
  fpga_novideo — the zeroed-video board outputs   (hw/board/snr_eval_novideo/chunk_*_outputs.npz,
                 produced by run_fpga.py --zero-video on the SAME audio)

Both use the SAME meta.npz (targets / scene_id / bin_idx / scene-count weights) and the SAME mixed-audio
baseline, scored with the IDENTICAL per-window metric + per-scene->bin->weighted aggregation as everywhere
else (reused from score_board_snr_bins). The fpga_video arm reproduces the stored board row
(4.59/1.615/0.735) — built-in sanity check.

Writes hw/board/snr_eval/board_video_ablation_results.json {fpga_video, fpga_novideo} and prints a per-bin
table with the (no-video - with-video) deltas for SI-SDR / PESQ / STOI.

  python tools/score_board_novideo.py
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from score_board_snr_bins import metric_one, aggregate


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", default=str(REPO / "hw/board/snr_eval"), help="with-video dir (meta + outputs + windows)")
    ap.add_argument("--nvdir", default=str(REPO / "hw/board/snr_eval_novideo"), help="zeroed-video outputs dir")
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    vd = Path(args.vdir); nvd = Path(args.nvdir)

    meta = np.load(vd / "meta.npz", allow_pickle=True)
    tgt = meta["target"]; sid = meta["scene_id"]; binx = meta["bin_idx"]
    chunk = meta["chunk"]; row = meta["row"]; labels = meta["bin_labels"]; full = meta["full_bin_scene_counts"]
    nch = int(meta["nchunks"]); N = len(sid)

    enh_v = np.zeros((N, tgt.shape[1]), dtype=np.float32)
    enh_nv = np.zeros_like(enh_v); mix = np.zeros_like(enh_v)
    missing = []
    for c in range(nch):
        ov = np.load(vd / f"chunk_{c:03d}_outputs.npz")["audio_out"].astype(np.float32) / 32768.0
        w = np.load(vd / f"chunk_{c:03d}_windows.npz")["audio_in"].astype(np.float32) / 32768.0
        nvp = nvd / f"chunk_{c:03d}_outputs.npz"
        if not nvp.exists():
            missing.append(c); continue
        onv = np.load(nvp)["audio_out"].astype(np.float32) / 32768.0
        for wi in np.where(chunk == c)[0]:
            enh_v[wi] = ov[row[wi]]; enh_nv[wi] = onv[row[wi]]; mix[wi] = w[row[wi]]
    if missing:
        print(f"ERROR: zeroed-video outputs missing for chunks {missing} — run the board first "
              f"(not fabricating). Aborting."); return 1

    print(f"scoring {N} windows x2 arms (with-video / zeroed-video), {args.nproc} procs ...", flush=True)
    pool = mp.Pool(args.nproc)
    pw_v, pw_nv = {}, {}
    for key, vals in pool.imap_unordered(metric_one, ((i, enh_v[i], mix[i], tgt[i]) for i in range(N)), chunksize=8):
        pw_v[key] = vals
    for key, vals in pool.imap_unordered(metric_one, ((i, enh_nv[i], mix[i], tgt[i]) for i in range(N)), chunksize=8):
        pw_nv[key] = vals
    pool.close(); pool.join()

    res_v = aggregate(pw_v, sid, binx, labels, full, "FPGA with video")
    res_nv = aggregate(pw_nv, sid, binx, labels, full, "FPGA NO video (zeroed)")
    out = {"fpga_video": res_v, "fpga_novideo": res_nv}
    rj = vd / "board_video_ablation_results.json"
    json.dump(out, open(rj, "w", encoding="utf-8"), indent=2)

    def cell(pv, pn, m):
        a = pv[m]; b = pn[m]
        if a is None or b is None: return "    n/a"
        return f"{a:6.2f}->{b:6.2f}({b-a:+5.2f})" if m == "si_e" else f"{a:5.3f}->{b:5.3f}({b-a:+5.3f})"

    print("\n========== ON-CHIP VISUAL ABLATION — per SNR bin (FPGA: with video -> video zeroed) ==========")
    print(f"{'bin':28s}{'n':>4s}   {'SI-SDR (dB)':>22s}   {'PESQ-WB':>21s}   {'STOI':>21s}")
    for b in sorted(res_v["per_bin"], key=int):
        pv = res_v["per_bin"][b]; pn = res_nv["per_bin"][b]
        if pv["si_e"] is None: continue
        print(f"{pv['label']:28s}{pv['n_scenes_scored']:>4d}   "
              f"{cell(pv, pn, 'si_e'):>22s}   {cell(pv, pn, 'p_e'):>21s}   {cell(pv, pn, 's_e'):>21s}")
    wv = res_v["overall"]["weighted"]; wn = res_nv["overall"]["weighted"]
    print("-" * 110)
    print(f"{'WEIGHTED (by bin scenes)':28s}{'':>4s}   "
          f"{wv['si_e']:6.2f}->{wn['si_e']:6.2f}({wn['si_e']-wv['si_e']:+5.2f})   "
          f"{wv['p_e']:5.3f}->{wn['p_e']:5.3f}({wn['p_e']-wv['p_e']:+5.3f})   "
          f"{wv['s_e']:5.3f}->{wn['s_e']:5.3f}({wn['s_e']-wv['s_e']:+5.3f})")
    print(f"\nwith-video weighted: SI-SDR {wv['si_e']:.3f} / PESQ {wv['p_e']:.3f} / STOI {wv['s_e']:.3f}"
          f"   (stored board fpga = 4.592 / 1.615 / 0.735 — should match)")
    print(f"saved -> {rj}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
