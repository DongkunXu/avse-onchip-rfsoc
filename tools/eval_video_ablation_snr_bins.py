"""eval_video_ablation_snr_bins.py — VISUAL ABLATION (pure-Python FP32) on the SAME per-SNR-bin scene set.

Goal: quantify the visual modality's contribution. For each window of the established 665-scene SNR-bin
set (hw/board/snr_eval/meta.npz), forward the FP32 model (experiments/p2-c7-full/best.pt) TWICE with the
SAME audio:
  (a) fp32_video   — real video frames (the normal model)
  (b) fp32_novideo — video frames zeroed (torch.zeros_like) = "black screen", lip motion removed

Both arms are scored with the IDENTICAL per-window metric + per-scene->per-bin->scene-count-weighted
aggregation used everywhere else in the project (reused from score_board_snr_bins). The with-video arm
reproduces the stored fp32 row (5.22/1.712/0.750) as a built-in sanity check.

Honest: nothing fabricated. If best.pt is missing, aborts. Writes video_ablation_results.json
{fp32_video, fp32_novideo} and prints a per-bin table with the (no-video - with-video) deltas for all
three metrics (SI-SDR / PESQ / STOI) so the contribution is judged from all three, not one.

  python tools/eval_video_ablation_snr_bins.py --dir hw/board/snr_eval --ckpt experiments/p2-c7-full/best.pt
"""
import argparse, os, sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from score_board_snr_bins import metric_one, aggregate  # identical metrics + aggregation


def main():
    import multiprocessing as mp
    import contextlib
    import torch
    from avse.config import Config
    from avse.data import AVSEDataset
    from avse.models import ConvTasNetAVSE

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "hw/board/snr_eval"))
    ap.add_argument("--ckpt", default=str(REPO / "experiments/p2-c7-full/best.pt"))
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    d = Path(args.dir)
    if not Path(args.ckpt).exists():
        print(f"ERROR: FP32 checkpoint not found: {args.ckpt} (not fabricating — aborting)"); return 1

    meta = np.load(d / "meta.npz", allow_pickle=True)
    sid_w = meta["scene_id"]; bin_w = meta["bin_idx"]
    labels = meta["bin_labels"]; full = meta["full_bin_scene_counts"]
    seen = {}
    for s, b in zip(sid_w, bin_w):
        if s not in seen:
            seen[s] = int(b)
    scenes = list(seen.keys())
    print(f"VIDEO ABLATION on {len(scenes)} scenes (same set as the board / FP32 run)", flush=True)

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvTasNetAVSE(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        ds = AVSEDataset(root_dir=cfg.data.root_dir, split="dev", config=cfg,
                         cache_dir=str(REPO / ".dataset_cache"))
    scene_wins = defaultdict(list)
    for i, w in enumerate(ds.windows):
        scene_wins[w["scene_id"]].append(i)

    items_v, items_nv = [], []          # (win_key, enh, mix, tgt) for each arm
    sid_arr, bin_arr = {}, {}
    buf, keys = [], []
    gi = 0

    def flush():
        nonlocal buf, keys
        if not buf:
            return
        vid = torch.stack([b[0] for b in buf]).to(device)
        mix = torch.stack([b[1] for b in buf]).to(device)
        with torch.no_grad():
            est_v = model({"video_frames": vid, "mixed_audio": mix}).cpu().numpy()
            est_nv = model({"video_frames": torch.zeros_like(vid), "mixed_audio": mix}).cpu().numpy()
        for k, (vv, mm, tt) in enumerate(buf):
            key = keys[k]
            m_np = mm.numpy()[0]; t_np = tt.numpy()[0]
            items_v.append((key, est_v[k, 0].copy(), m_np.copy(), t_np.copy()))
            items_nv.append((key, est_nv[k, 0].copy(), m_np.copy(), t_np.copy()))
        buf, keys = [], []

    t0 = time.time(); done = 0
    for sid in scenes:
        for wi in scene_wins[sid]:
            it = ds[wi]
            sid_arr[gi] = sid; bin_arr[gi] = seen[sid]
            buf.append((it["video_frames"], it["mixed_audio"], it["target_audio"]))
            keys.append(gi); gi += 1
            if len(buf) >= args.batch:
                flush()
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(scenes)} scenes forwarded ({gi} win, "
                  f"{gi/(time.time()-t0):.0f} win/s)", flush=True)
    flush()

    N = gi
    sid_np = np.empty(N, dtype=object); bin_np = np.zeros(N, dtype=np.int16)
    for k in range(N):
        sid_np[k] = sid_arr[k]; bin_np[k] = bin_arr[k]

    print(f"scoring {N} windows x2 arms, {args.nproc} procs ...", flush=True)
    pool = mp.Pool(args.nproc)
    pw_v, pw_nv = {}, {}
    for key, vals in pool.imap_unordered(metric_one, items_v, chunksize=8):
        pw_v[key] = vals
    for key, vals in pool.imap_unordered(metric_one, items_nv, chunksize=8):
        pw_nv[key] = vals
    pool.close(); pool.join()

    res_v = aggregate(pw_v, sid_np, bin_np, labels, full, "FP32 with video")
    res_nv = aggregate(pw_nv, sid_np, bin_np, labels, full, "FP32 NO video (zeroed)")

    out = {"fp32_video": res_v, "fp32_novideo": res_nv}
    rj = d / "video_ablation_results.json"
    json.dump(out, open(rj, "w", encoding="utf-8"), indent=2)

    # ---- report: per bin, all three metrics, with deltas (no-video - with-video) ----
    def row(pb_v, pb_nv, m):
        a = pb_v[m]; b = pb_nv[m]
        if a is None or b is None:
            return "    n/a"
        return f"{a:6.2f}->{b:6.2f}({b-a:+5.2f})" if m == "si_e" else f"{a:5.3f}->{b:5.3f}({b-a:+5.3f})"

    print("\n================ VISUAL ABLATION — per SNR bin (with video -> video zeroed) ================")
    print(f"{'bin':28s}{'n':>4s}   {'SI-SDR (dB)':>22s}   {'PESQ-WB':>21s}   {'STOI':>21s}")
    for b in sorted(res_v["per_bin"], key=int):
        pv = res_v["per_bin"][b]; pn = res_nv["per_bin"][b]
        if pv["si_e"] is None:
            continue
        print(f"{pv['label']:28s}{pv['n_scenes_scored']:>4d}   "
              f"{row(pv, pn, 'si_e'):>22s}   {row(pv, pn, 'p_e'):>21s}   {row(pv, pn, 's_e'):>21s}")
    wv = res_v["overall"]["weighted"]; wn = res_nv["overall"]["weighted"]
    print("-" * 110)
    print(f"{'WEIGHTED (by bin scenes)':28s}{'':>4s}   "
          f"{wv['si_e']:6.2f}->{wn['si_e']:6.2f}({wn['si_e']-wv['si_e']:+5.2f})   "
          f"{wv['p_e']:5.3f}->{wn['p_e']:5.3f}({wn['p_e']-wv['p_e']:+5.3f})   "
          f"{wv['s_e']:5.3f}->{wn['s_e']:5.3f}({wn['s_e']-wv['s_e']:+5.3f})")
    print(f"\nwith-video weighted: SI-SDR {wv['si_e']:.3f} / PESQ {wv['p_e']:.3f} / STOI {wv['s_e']:.3f}"
          f"   (stored fp32 = 5.216 / 1.712 / 0.750 — should match)")
    print(f"merged -> {rj}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
