"""eval_full_dev.py — definitive full-split quality evaluation for the time-domain AVSE candidates.

Mirrors the reference deployment's evaluation protocol (../UNet-AVSE-Vitis, distilled in
`test reference/scripts/evaluate_snr_bins.py`): for every scene, slide the fixed analysis window over
the whole utterance, score each window's enhanced output AND the mixed-input baseline with
STOI / PESQ-WB / SI-SDR, average per scene, then average over ALL scenes (scene-weighted). This is
far more reliable than the 200-window subset used during training (~25 scenes), which is noisy.

The metric functions are byte-identical to the reference harness (`avse.metrics`). The model is run
with the **training-consistent per-window 0.8/|mixed|.max() normalization**, which is also the
on-chip streaming-deployment normalization, so the numbers reflect how the model is actually used.

Efficiency: the GPU does the batched model forward; the PESQ/STOI/SI-SDR bottleneck is parallelised
across a CPU process Pool. The top of this module stays light (numpy + avse.metrics, NO torch) so
spawned Pool workers do not re-import torch; all heavy work lives under the __main__ guard.

Usage:
    python tools/eval_full_dev.py --ckpt experiments/p2-c7-full/best.pt
    python tools/eval_full_dev.py --ckpt experiments/p2-c7-hq/best.pt --split dev --nproc 16
"""
import os
import sys
import numpy as np

from avse.metrics import si_sdr, pesq_wb, stoi_score


def metric_one(args):
    """Per-window metrics for the enhanced output and the mixed-input baseline. Runs in a Pool worker."""
    sid, e, t, m = args
    return sid, (
        si_sdr(e, t), si_sdr(m, t),          # SI-SDR: enhanced, mixed
        pesq_wb(t, e), pesq_wb(t, m),        # PESQ-WB: enhanced, mixed
        stoi_score(t, e), stoi_score(t, m),  # STOI:    enhanced, mixed
    )


def main() -> int:
    import argparse
    import json
    import time
    import contextlib
    import multiprocessing as mp
    from collections import defaultdict
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from avse.config import Config
    from avse.data import AVSEDataset, AVSESceneStreamDataset
    from avse.models import ConvTasNetAVSE, StreamingTCNAVSE

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    REPO = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a state_dict (.pt), e.g. experiments/<exp>/best.pt")
    ap.add_argument("--model", choices=["c7", "c2"], default="c7")
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None, help="output json (default: <ckpt_dir>/full_<split>_eval.json)")
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = full split; >0 for a quick smoke")
    args = ap.parse_args()

    out_path = args.out or str(Path(args.ckpt).resolve().parent / f"full_{args.split}_eval.json")
    Model = {"c7": ConvTasNetAVSE, "c2": StreamingTCNAVSE}[args.model]

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        base = AVSEDataset(root_dir=cfg.data.root_dir, split=args.split, config=cfg,
                           cache_dir=str(REPO / ".dataset_cache"))
    ds = AVSESceneStreamDataset(base, shuffle=False, max_windows=args.max_windows)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)
    n_scenes, n_windows = len(ds.scene_ids), len(ds)
    print(f"ckpt={args.ckpt} | {args.split}: scenes={n_scenes} windows={n_windows} | "
          f"device={device} | pool={args.nproc}", flush=True)

    KEYS = ("si_e", "si_m", "p_e", "p_m", "s_e", "s_m")
    acc = defaultdict(lambda: {k: [] for k in KEYS})

    def collect(sid, vals):
        a = acc[sid]
        for k, v in zip(KEYS, vals):
            if v is not None:
                a[k].append(v)

    t0 = time.time()
    seen = 0
    chunk_cap = max(args.nproc * 120, args.batch)
    pool = mp.Pool(processes=args.nproc)
    buf = []

    def flush():
        for sid, vals in pool.imap_unordered(metric_one, buf, chunksize=16):
            collect(sid, vals)
        buf.clear()

    with torch.no_grad():
        for batch in loader:
            vid = batch["video_frames"].to(device)
            mix = batch["mixed_audio"].to(device)
            est = model({"video_frames": vid, "mixed_audio": mix}).cpu().numpy()
            tgt = batch["target_audio"].numpy()
            mixc = batch["mixed_audio"].numpy()
            sids = batch["scene_id"]
            for i in range(est.shape[0]):
                buf.append((sids[i], est[i, 0].copy(), tgt[i, 0].copy(), mixc[i, 0].copy()))
            seen += est.shape[0]
            if len(buf) >= chunk_cap:
                flush()
                print(f"  {seen}/{n_windows} windows | {seen/(time.time()-t0):.0f} win/s | "
                      f"{len(acc)} scenes", flush=True)
    if buf:
        flush()
    pool.close()
    pool.join()

    per_scene = {sid: {k: (float(np.mean(a[k])) if a[k] else None) for k in KEYS}
                 for sid, a in acc.items()}

    def overall(metric):
        vals = [v[metric] for v in per_scene.values() if v[metric] is not None]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    res = {k: overall(k) for k in KEYS}
    summary = {
        "ckpt": args.ckpt, "split": args.split, "scenes": n_scenes, "windows": n_windows,
        "scenes_scored": len(per_scene), "sec": round(time.time() - t0, 1),
        "enhanced": {"si_sdr": res["si_e"][0], "pesq_wb": res["p_e"][0], "stoi": res["s_e"][0]},
        "mixed_input": {"si_sdr": res["si_m"][0], "pesq_wb": res["p_m"][0], "stoi": res["s_m"][0]},
        "improvement": {
            "si_sdr": (res["si_e"][0] - res["si_m"][0]) if res["si_e"][0] is not None else None,
            "pesq_wb": (res["p_e"][0] - res["p_m"][0]) if res["p_e"][0] is not None else None,
            "stoi": (res["s_e"][0] - res["s_m"][0]) if res["s_e"][0] is not None else None,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_scene": per_scene}, f, indent=2)

    print("\n================  FULL-%s RESULT  ================" % args.split.upper(), flush=True)
    print(f"scenes scored: {summary['scenes_scored']}/{n_scenes} | {summary['sec']:.0f}s")
    print(f"{'':14s}{'SI-SDR(dB)':>12s}{'PESQ-WB':>10s}{'STOI':>8s}")
    for row in ("enhanced", "mixed_input", "improvement"):
        s = summary[row]
        sign = "+" if row == "improvement" else ""
        print(f"{row:14s}{s['si_sdr']:>{12}.3f}{s['pesq_wb']:>{10}.3f}{s['stoi']:>{8}.3f}"
              if sign == "" else
              f"{row:14s}{s['si_sdr']:>+12.3f}{s['pesq_wb']:>+10.3f}{s['stoi']:>+8.3f}")
    print(f"saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
