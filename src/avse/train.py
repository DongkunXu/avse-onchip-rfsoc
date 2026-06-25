"""train.py — Phase-2 training harness for the time-domain AVSE candidates (C7, C2).

Plain-PyTorch loop (transparent, controllable) reusing the migrated loss + metrics. Trains a chosen
model on LRS3, evaluates SI-SDR / PESQ / STOI, and writes everything to experiments/<exp_id>/ so each
run is a self-contained, reproducible point on the quality-vs-working-set Pareto.

Examples:
    # fast smoke (validate the loop): tiny subset, 1 epoch, CPU/GPU auto
    python -m avse.train --model c7 --quick
    # a real short run
    python -m avse.train --model c7 --epochs 8 --max-train-windows 6000 --workers 6
"""
from __future__ import annotations

import sys, os, json, time, argparse, random
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from avse.config import Config
from avse.data import AVSEDataset
from avse.losses import ImprovedAVSELoss
from avse.metrics import si_sdr, pesq_wb, stoi_score
from avse.models import ConvTasNetAVSE, StreamingTCNAVSE

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "experiments"

MODELS = {
    "c7": ("ConvTasNetAVSE (mask)", ConvTasNetAVSE),
    "c2": ("StreamingTCNAVSE (mapping)", StreamingTCNAVSE),
}


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_loader(cfg, split, bs, workers, max_windows=0, shuffle=True, seed=42, prefetch=2):
    ds = AVSEDataset(root_dir=cfg.data.root_dir, split=split, config=cfg)
    if max_windows and max_windows < len(ds):
        # Subset the windows list IN-PLACE (not Subset-over-full): each DataLoader worker pickles the
        # dataset, so carrying only the needed windows (not all ~250k) keeps per-worker memory small.
        # Important on this 32 GB host — the full-dataset copy + large video tensors in shared memory
        # exhausted the Windows commit limit (error 1455) on long runs.
        g = random.Random(seed)
        idx = list(range(len(ds))); g.shuffle(idx)
        ds.windows = [ds.windows[i] for i in idx[:max_windows]]
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=(workers > 0), drop_last=shuffle,
                      persistent_workers=(workers > 0),
                      prefetch_factor=(prefetch if workers > 0 else None))


@torch.no_grad()
def evaluate(model, loader, device, max_pesq_stoi=120):
    """SI-SDR over all eval samples; PESQ/STOI over the first `max_pesq_stoi` (they are slow)."""
    model.eval()
    sis, pesqs, stois = [], [], []
    n_perc = 0
    for batch in loader:
        vid = batch["video_frames"].to(device)
        mix = batch["mixed_audio"].to(device)
        tgt = batch["target_audio"]
        est = model({"video_frames": vid, "mixed_audio": mix}).cpu()
        for i in range(est.shape[0]):
            e = est[i, 0].numpy(); t = tgt[i, 0].numpy()
            sis.append(si_sdr(e, t))
            if n_perc < max_pesq_stoi:
                p = pesq_wb(t, e); s = stoi_score(t, e)
                if p is not None: pesqs.append(p)
                if s is not None: stois.append(s)
                n_perc += 1
    def m(a): return float(np.mean(a)) if a else None
    return {"si_sdr": m(sis), "pesq_wb": m(pesqs), "stoi": m(stois), "n": len(sis)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="dev")
    ap.add_argument("--max-train-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--max-val-windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny subset + 1 epoch to validate the loop")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 1
        args.train_split = "dev"; args.max_train_windows = 64
        args.max_val_windows = 32; args.batch = 8; args.workers = 0

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config.from_yaml(args.config)

    name, Model = MODELS[args.model]
    model = Model(cfg).to(device)
    ws = model.deployable_working_set_elems()
    exp_id = args.exp_id or f"p2-{args.model}-{'quick' if args.quick else 'run'}"
    out = EXP_DIR / exp_id; out.mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} | {name} ===")
    print(f"device={device} | params={model.num_params():,} | "
          f"deployable working set={ws:,} elems ({ws*2/1e6:.3f} MB audio path)")

    train_loader = build_loader(cfg, args.train_split, args.batch, args.workers,
                                args.max_train_windows, shuffle=True, seed=args.seed)
    val_loader = build_loader(cfg, args.val_split, args.batch, args.workers,
                              args.max_val_windows, shuffle=False, seed=args.seed)
    print(f"train batches={len(train_loader)} | val batches={len(val_loader)}")

    criterion = ImprovedAVSELoss(cfg.loss).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history = []
    best_sisdr = -1e9
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        run = 0.0; nb = 0; t0 = time.time()
        for bi, batch in enumerate(train_loader):
            vid = batch["video_frames"].to(device, non_blocking=True)
            mix = batch["mixed_audio"].to(device, non_blocking=True)
            tgt = batch["target_audio"].to(device, non_blocking=True)
            est = model({"video_frames": vid, "mixed_audio": mix})
            losses = criterion(est, tgt)
            opt.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += losses["total_loss"].item(); nb += 1
            if bi % 20 == 0:
                print(f"  ep{ep} b{bi}/{len(train_loader)} loss={run/nb:.4f}", flush=True)
        val = evaluate(model, val_loader, device)
        rec = {"epoch": ep, "train_loss": run / max(nb, 1), "lr": opt.param_groups[0]["lr"],
               "val": val, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"ep{ep}: train_loss={rec['train_loss']:.4f} lr={rec['lr']:.2e} | "
              f"val SI-SDR={val['si_sdr']:.2f}dB PESQ={val['pesq_wb']} STOI={val['stoi']} "
              f"({rec['sec']}s)", flush=True)
        # Checkpoint every epoch so a crash (e.g. a long-run DataLoader memory error) doesn't lose
        # progress; also keep the best-by-SI-SDR snapshot. metrics.json is rewritten each epoch too.
        torch.save(model.state_dict(), out / "checkpoint.pt")
        if val["si_sdr"] is not None and val["si_sdr"] >= best_sisdr:
            best_sisdr = val["si_sdr"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "metrics.json").write_text(json.dumps(
            {"exp_id": exp_id, "model": args.model, "model_name": name,
             "params": model.num_params(), "deployable_working_set_elems": ws,
             "deployable_working_set_mb": round(ws * 2 / 1e6, 4), "args": vars(args),
             "device": device, "history": history,
             "final_val": history[-1]["val"], "best_si_sdr": best_sisdr}, indent=2), encoding="utf-8")
        sched.step()

    torch.save(model.state_dict(), out / "checkpoint.pt")
    result = {
        "exp_id": exp_id, "model": args.model, "model_name": name,
        "params": model.num_params(),
        "deployable_working_set_elems": ws,
        "deployable_working_set_mb": round(ws * 2 / 1e6, 4),
        "args": vars(args), "device": device,
        "total_sec": round(time.time() - t_start, 1),
        "history": history, "final_val": history[-1]["val"] if history else None,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}/metrics.json + checkpoint.pt")
    print(f"FINAL val: {result['final_val']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
