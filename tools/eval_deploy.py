"""eval_deploy.py — deployment-accurate full-dev quality of the C7 on-chip AVSE.

Same protocol as `tools/eval_full_dev.py` (same dataset, same per-scene sliding windows, same scene-weighted
STOI / PESQ-WB / SI-SDR, same metric functions) but the forward pass is the **fixed-point emulator**
(`tools/c7_fixedpoint.py`) built from the exported deploy weights, so the numbers reflect what the int16
on-chip design actually computes.

Validation of the chain: run with `--precision fp --mask sigmoid` -> the emulator stands in for the trained
PyTorch model and must reproduce the committed FP32 number (~5.40), confirming the emulator + eval protocol
match `eval_full_dev`. Then `--precision int16 --mask hardsigmoid` gives the deployment number.

Usage:
    python tools/eval_deploy.py --precision int16 --mask hardsigmoid
    python tools/eval_deploy.py --precision fp --mask sigmoid        # protocol/chain check vs 5.40
"""
import os
import sys
import numpy as np

from avse.metrics import si_sdr, pesq_wb, stoi_score


def metric_one(args):
    """Per-window metrics for enhanced + mixed-input baseline. Runs in a Pool worker (no torch import)."""
    sid, e, t, m = args
    return sid, (
        si_sdr(e, t), si_sdr(m, t),
        pesq_wb(t, e), pesq_wb(t, m),
        stoi_score(t, e), stoi_score(t, m),
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

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from c7_fixedpoint import C7FixedPoint

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    REPO = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO / "experiments/p2-c7-full/deploy_weights.npz"))
    ap.add_argument("--precision", choices=["int16", "fp"], default="int16")
    ap.add_argument("--mask", choices=["hardsigmoid", "sigmoid"], default="hardsigmoid")
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nproc", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = full split; >0 for a quick smoke")
    args = ap.parse_args()

    tag = f"{args.precision}_{args.mask}"
    out_path = args.out or str(Path(args.npz).resolve().parent / f"deploy_{args.split}_eval_{tag}.json")

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emu = C7FixedPoint(args.npz, device=device, precision=args.precision, mask=args.mask)

    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        base = AVSEDataset(root_dir=cfg.data.root_dir, split=args.split, config=cfg,
                           cache_dir=str(REPO / ".dataset_cache"))
    ds = AVSESceneStreamDataset(base, shuffle=False, max_windows=args.max_windows)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)
    n_scenes, n_windows = len(ds.scene_ids), len(ds)
    print(f"deploy-eval [{tag}] | {args.split}: scenes={n_scenes} windows={n_windows} | "
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
            est = emu.forward(mix, vid).cpu().numpy()
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
        "tag": tag, "precision": args.precision, "mask": args.mask, "npz": args.npz,
        "split": args.split, "scenes": n_scenes, "windows": n_windows,
        "scenes_scored": len(per_scene), "sec": round(time.time() - t0, 1),
        "enhanced": {"si_sdr": res["si_e"][0], "pesq_wb": res["p_e"][0], "stoi": res["s_e"][0]},
        "mixed_input": {"si_sdr": res["si_m"][0], "pesq_wb": res["p_m"][0], "stoi": res["s_m"][0]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_scene": per_scene}, f, indent=2)

    print(f"\n========  DEPLOY-{args.split.upper()} [{tag}]  ========", flush=True)
    print(f"scenes scored: {summary['scenes_scored']}/{n_scenes} | {summary['sec']:.0f}s")
    print(f"{'':14s}{'SI-SDR(dB)':>12s}{'PESQ-WB':>10s}{'STOI':>8s}")
    for row in ("enhanced", "mixed_input"):
        s = summary[row]
        print(f"{row:14s}{s['si_sdr']:>12.3f}{s['pesq_wb']:>10.3f}{s['stoi']:>8.3f}")
    print(f"saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
