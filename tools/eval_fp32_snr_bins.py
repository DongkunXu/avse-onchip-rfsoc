"""eval_fp32_snr_bins.py — FP32 (original best.pt) per-SNR-bin eval on the SAME scenes as the board run.

Reads meta.npz (the exact sampled scenes + bins from prep_board_snr_bins.py), runs the FP32 model with
FULL-PRECISION inputs via AVSEDataset (the eval_full_dev path), scores per scene -> per bin -> scene-count-
weighted, and MERGES the result into snr_bin_results.json under key "fp32" (so the plot has FP32 / emulator /
FPGA together). This is the FP32 upper bound; the FPGA and emulator use int16 inputs, FP32 uses float inputs
(all on the identical scene set).

  python tools/eval_fp32_snr_bins.py --dir hw/board/snr_eval --ckpt experiments/p2-c7-full/best.pt
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from score_board_snr_bins import metric_one, aggregate  # reuse identical metrics + aggregation


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
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    d = Path(args.dir)
    if not Path(args.ckpt).exists():
        print(f"ERROR: FP32 checkpoint not found: {args.ckpt} (not fabricating — aborting)"); return 1

    meta = np.load(d / "meta.npz", allow_pickle=True)
    sid_w = meta["scene_id"]; bin_w = meta["bin_idx"]
    labels = meta["bin_labels"]; full = meta["full_bin_scene_counts"]
    # unique sampled scenes (preserve a stable order) + their bin
    seen = {}
    for s, b in zip(sid_w, bin_w):
        if s not in seen:
            seen[s] = int(b)
    scenes = list(seen.keys())
    print(f"FP32 eval on {len(scenes)} scenes (same set as the board run)", flush=True)

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvTasNetAVSE(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        ds = AVSEDataset(root_dir=cfg.data.root_dir, split="dev", config=cfg,
                         cache_dir=str(REPO / ".dataset_cache"))
    from collections import defaultdict
    scene_wins = defaultdict(list)
    for i, w in enumerate(ds.windows):
        scene_wins[w["scene_id"]].append(i)

    # forward all sampled windows; build the (key, enhanced, mixed, target) list for the shared scorer
    items = []          # (global_win_key, enh, mix, tgt)
    sid_arr = {}; bin_arr = {}
    buf, meta_b = [], []
    gi = 0

    def flush_fwd():
        nonlocal buf, meta_b
        if not buf:
            return
        vid = torch.stack([b[0] for b in buf]).to(device)
        mix = torch.stack([b[1] for b in buf]).to(device)
        with torch.no_grad():
            est = model({"video_frames": vid, "mixed_audio": mix}).cpu().numpy()
        for k, (vv, mm, tt) in enumerate(buf):
            key = meta_b[k]
            items.append((key, est[k, 0].copy(), mm.numpy()[0].copy(), tt.numpy()[0].copy()))
        buf, meta_b = [], []

    t0 = __import__("time").time()
    done = 0
    for sid in scenes:
        for wi in scene_wins[sid]:
            it = ds[wi]
            sid_arr[gi] = sid; bin_arr[gi] = seen[sid]
            buf.append((it["video_frames"], it["mixed_audio"], it["target_audio"]))
            meta_b.append(gi); gi += 1
            if len(buf) >= args.batch:
                flush_fwd()
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(scenes)} scenes forwarded ({gi} win, {gi/(__import__('time').time()-t0):.0f} win/s)", flush=True)
    flush_fwd()

    # build per-window arrays expected by aggregate (keys are 0..N-1)
    N = gi
    sid_np = np.empty(N, dtype=object); bin_np = np.zeros(N, dtype=np.int16)
    for k in range(N):
        sid_np[k] = sid_arr[k]; bin_np[k] = bin_arr[k]
    print(f"scoring {N} FP32 windows, {args.nproc} procs ...", flush=True)
    pool = mp.Pool(args.nproc)
    per_win = {}
    for key, vals in pool.imap_unordered(metric_one, items, chunksize=8):
        per_win[key] = vals
    pool.close(); pool.join()
    res = aggregate(per_win, sid_np, bin_np, labels, full, "FP32 (sw, float in)")

    # merge into snr_bin_results.json
    rj = d / "snr_bin_results.json"
    out = json.load(open(rj)) if rj.exists() else {}
    out["fp32"] = res
    json.dump(out, open(rj, "w", encoding="utf-8"), indent=2)

    print(f"\n================ {res['tag']} — per SNR bin ================")
    print(f"{'bin':28s}{'n':>4s}{'SI-SDR':>9s}{'PESQ':>8s}{'STOI':>7s}")
    for b in sorted(res["per_bin"], key=int):
        pb = res["per_bin"][b]
        if pb["si_e"] is None: continue
        print(f"{pb['label']:28s}{pb['n_scenes_scored']:>4d}{pb['si_e']:>9.2f}{pb['p_e']:>8.3f}{pb['s_e']:>7.3f}")
    w = res["overall"]["weighted"]
    print(f"{'WEIGHTED (by bin scenes)':28s}{'':>4s}{w['si_e']:>9.2f}{w['p_e']:>8.3f}{w['s_e']:>7.3f}")
    print(f"merged -> {rj}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
