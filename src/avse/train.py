"""train.py — training harness for the time-domain AVSE candidates (C7, C2).

Plain-PyTorch loop reusing the migrated loss + metrics. Trains a chosen model on LRS3, evaluates
SI-SDR / PESQ / STOI, and writes everything to experiments/<exp_id>/ (checkpoint, best, metrics.json,
trend.png) so each run is self-contained and resumable.

Output is ASCII-only with two live progress bars (epochs on top, batches below) that update in place.

Examples:
    # fast smoke (validate the loop)
    python -m avse.train --model c7 --quick
    # the definitive full-data run (all train windows, 80 epochs, early-stop after 5 no-improve)
    python -m avse.train --model c7 --exp-id p2-c7-full --epochs 80 --early-stop-patience 5 \
        --batch 32 --workers 4 --prefetch 4 --max-train-windows 0 --lr 5e-4
    # resume an interrupted run from its last checkpoint
    python -m avse.train --model c7 --exp-id p2-c7-full ... --resume
"""
from __future__ import annotations

import sys, os, json, time, argparse, random, contextlib
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def build_loader(cfg, split, bs, workers, max_windows=0, shuffle=True, seed=42, prefetch=4):
    # Redirect only STDOUT to a utf-8 devnull (kills the dataset's emoji/CJK status prints); leave
    # STDERR so its ASCII tqdm scan bars stay visible — the full 'train' split is ~315k windows and
    # the scan takes a few minutes, so live progress is what stops it looking hung.
    note = " (~315k windows, scan takes a few minutes)" if split == "train" else ""
    print(f"[data] loading '{split}' split{note}...", flush=True)
    t0 = time.time()
    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        ds = AVSEDataset(root_dir=cfg.data.root_dir, split=split, config=cfg,
                         cache_dir=str(REPO / ".dataset_cache"))
    if max_windows and max_windows < len(ds):
        # Subset the windows list IN-PLACE (not Subset-over-full): each DataLoader worker pickles the
        # dataset, so carrying only the needed windows keeps per-worker memory small on this 32 GB host
        # (the full-dataset copy + large video tensors in shared memory exhausted the commit limit).
        g = random.Random(seed)
        idx = list(range(len(ds))); g.shuffle(idx)
        ds.windows = [ds.windows[i] for i in idx[:max_windows]]
    print(f"[data] '{split}': {len(ds)} windows in {time.time()-t0:.1f}s", flush=True)
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


def save_trend(history, path):
    """Per-epoch quality trend (SI-SDR / PESQ / STOI vs epoch). Best-effort; never fatal."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        def series(key):
            xs = [h["epoch"] for h in history if h["val"].get(key) is not None]
            ys = [h["val"][key] for h in history if h["val"].get(key) is not None]
            return xs, ys
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for a, key, title, col in zip(
                ax, ["si_sdr", "pesq_wb", "stoi"], ["SI-SDR (dB)", "PESQ-WB", "STOI"],
                ["tab:blue", "tab:orange", "tab:green"]):
            xs, ys = series(key)
            a.plot(xs, ys, marker="o", color=col)
            a.set_title(title); a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    except Exception:
        pass


def write_metrics(out, base, history, best):
    (out / "metrics.json").write_text(json.dumps(
        {**base, "history": history, "final_val": history[-1]["val"] if history else None,
         "best_si_sdr": best}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="stop after this many epochs without val SI-SDR improvement (0 = off)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="dev")
    ap.add_argument("--max-train-windows", type=int, default=0, help="0 = all (full data)")
    ap.add_argument("--max-val-windows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from experiments/<exp_id>/checkpoint.pt")
    ap.add_argument("--quick", action="store_true", help="tiny subset + 2 epochs to validate the loop")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2; args.early_stop_patience = 0
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
    base = {"exp_id": exp_id, "model": args.model, "model_name": name,
            "params": model.num_params(), "deployable_working_set_elems": ws,
            "deployable_working_set_mb": round(ws * 2 / 1e6, 4), "args": vars(args), "device": device}

    print(f"=== {exp_id} | {name} ===")
    print(f"device={device} | params={model.num_params():,} | "
          f"deployable working set={ws:,} elems ({ws*2/1e6:.3f} MB audio path)")

    criterion = ImprovedAVSELoss(cfg.loss).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history, best_sisdr, start_ep, no_improve = [], -1e9, 0, 0
    ckpt_path = out / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        history = ck.get("history", []); best_sisdr = ck.get("best_sisdr", -1e9)
        start_ep = ck.get("epoch", -1) + 1; no_improve = ck.get("no_improve", 0)
        print(f"[resume] from epoch {start_ep} (best SI-SDR {best_sisdr:.2f} dB)")

    train_loader = build_loader(cfg, args.train_split, args.batch, args.workers,
                                args.max_train_windows, shuffle=True, seed=args.seed, prefetch=args.prefetch)
    val_loader = build_loader(cfg, args.val_split, args.batch, args.workers,
                              args.max_val_windows, shuffle=False, seed=args.seed, prefetch=args.prefetch)
    print(f"train batches={len(train_loader)} | val batches={len(val_loader)}\n", flush=True)

    t_start = time.time()
    epoch_bar = tqdm(range(start_ep, args.epochs), desc="epochs", position=0,
                     ascii=True, dynamic_ncols=True, leave=True)
    for ep in epoch_bar:
        model.train()
        run = 0.0; nb = 0; nan_skips = 0; t0 = time.time()
        bbar = tqdm(train_loader, desc=f"  ep{ep:02d}", position=1, ascii=True,
                    dynamic_ncols=True, leave=False)
        for batch in bbar:
            vid = batch["video_frames"].to(device, non_blocking=True)
            mix = batch["mixed_audio"].to(device, non_blocking=True)
            tgt = batch["target_audio"].to(device, non_blocking=True)
            est = model({"video_frames": vid, "mixed_audio": mix})
            losses = criterion(est, tgt)
            total = losses["total_loss"]
            # Guard: some LRS3 windows have an all-zero (silent) target, which makes the perceptual
            # PESQ/SI-SDR terms NaN. Skip such batches (no backward/step) so a NaN gradient never
            # poisons the weights. Rare -> negligible data loss; counted and reported per epoch.
            if not torch.isfinite(total):
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += total.item(); nb += 1
            bbar.set_postfix(loss=f"{run/nb:+.3f}", skip=nan_skips, refresh=False)
        bbar.close()

        val = evaluate(model, val_loader, device)
        rec = {"epoch": ep, "train_loss": run / max(nb, 1), "lr": opt.param_groups[0]["lr"],
               "val": val, "sec": round(time.time() - t0, 1), "nan_skips": nan_skips}
        history.append(rec)

        improved = val["si_sdr"] is not None and val["si_sdr"] > best_sisdr + 1e-4
        if improved:
            best_sisdr = val["si_sdr"]; no_improve = 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            no_improve += 1

        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best_sisdr": best_sisdr,
                    "no_improve": no_improve, "history": history}, ckpt_path)
        write_metrics(out, base, history, best_sisdr)
        save_trend(history, out / "trend.png")

        si = val["si_sdr"]; pe = val["pesq_wb"]; st = val["stoi"]
        epoch_bar.set_postfix(
            si_sdr=f"{si:+.2f}" if si is not None else "na",
            pesq=f"{pe:.3f}" if pe is not None else "na",
            stoi=f"{st:.3f}" if st is not None else "na",
            best=f"{best_sisdr:+.2f}", noimp=f"{no_improve}/{args.early_stop_patience}",
            refresh=True)
        epoch_bar.write(
            f"ep{ep:02d} | loss {rec['train_loss']:+.4f} lr {rec['lr']:.2e} | "
            f"SI-SDR {('%.2f' % si) if si is not None else 'na'} PESQ "
            f"{('%.3f' % pe) if pe is not None else 'na'} STOI "
            f"{('%.3f' % st) if st is not None else 'na'} | best {best_sisdr:+.2f} "
            f"| nan-skip {nan_skips} | {rec['sec']:.0f}s")

        sched.step()
        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            epoch_bar.write(f"[early-stop] no SI-SDR improvement for {no_improve} epochs -> stopping at ep{ep}")
            break
    epoch_bar.close()

    write_metrics(out, {**base, "total_sec": round(time.time() - t_start, 1)}, history, best_sisdr)
    print(f"\nsaved -> {out}/  (metrics.json, best.pt, checkpoint.pt, trend.png)")
    print(f"BEST val SI-SDR: {best_sisdr:+.2f} dB | final: {history[-1]['val'] if history else None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
