"""prep_board_snr_bins.py — PC-side: build a stratified per-SNR-bin subset of the dev split, quantized to
int16, chunked for the on-board FPGA run (the optimized bitstream, ~0.286 s/window).

Uses the ESTABLISHED reference binning: the 10 SNR bins (2.5 dB, -15..+10) from
`test reference/selection_manifest.json` (`bin_definitions` + `selection.dev` = the full dev set assigned
to bins). A fraction (default 20%) of each bin's scenes is sampled, seeded -> representative + comparable.

Protocol = our validated path (AVSEDataset windowing + per-window 0.8/|mixed|.max() normalization, the same
as eval_full_dev.py / prep_board_windows.py), so the FPGA numbers are consistent with the rest of the project.

Outputs under --out-dir:
  chunk_XXX_windows.npz  (audio_in [n,19200] int16, video_in [n,30,96,96] int16)  -> copy each to board
  meta.npz               (per-window target float32 [N,19200], scene_id, bin_idx, snr, chunk, row;
                          + bin_labels, bin_lo, bin_hi, full_bin_scene_counts for the weighted overall)

Usage:
  python tools/prep_board_snr_bins.py --frac 0.2 --seed 42 --chunk 500
  python tools/prep_board_snr_bins.py --frac 0.2 --max-scenes-per-bin 2 --out-dir hw/board/snr_smoke  # smoke
"""
import argparse, os, sys, json, contextlib
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def q_int16(x, frac, lo=-32768, hi=32767):
    q = np.floor(np.asarray(x, dtype=np.float64) * (2 ** frac))
    return np.clip(q, lo, hi).astype(np.int16)


def main():
    from avse.config import Config
    from avse.data import AVSEDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of scenes sampled per SNR bin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=500, help="windows per board chunk (board RAM)")
    ap.add_argument("--manifest", default=str(REPO / "test reference/selection_manifest.json"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "hw/board/snr_eval"))
    ap.add_argument("--max-scenes-per-bin", type=int, default=0, help="cap per bin (0=none); for smoke tests")
    args = ap.parse_args()
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

    man = json.load(open(args.manifest, encoding="utf-8"))
    bins = man["bin_definitions"]                       # [{index,label,low,high}]
    sel = man["selection"][args.split]                 # {label: [scene_ids]}
    nb = len(bins)

    cfg = Config.from_yaml(args.config)
    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        ds = AVSEDataset(root_dir=cfg.data.root_dir, split=args.split, config=cfg,
                         cache_dir=str(REPO / ".dataset_cache"))
    scene_wins = defaultdict(list)
    for i, w in enumerate(ds.windows):
        scene_wins[w["scene_id"]].append(i)
    snr_of = {s["scene"]: float(s["SNR"]) for s in json.load(
        open(Path(cfg.data.root_dir) / f"scenes.{args.split}.json", encoding="utf-8"))}

    rng = np.random.default_rng(args.seed)
    full_counts = np.zeros(nb, dtype=int)
    win_rows = []  # (global_window_idx, scene_id, bin_idx, snr)
    print(f"{args.split}: {nb} reference SNR bins; sampling frac={args.frac} seed={args.seed}")
    for b in bins:
        bi, lab = b["index"], b["label"]
        present = [s for s in sel[lab] if s in scene_wins]   # scenes with usable windows
        full_counts[bi] = len(present)
        if not present:
            print(f"  {lab}: 0 usable"); continue
        k = max(1, int(round(args.frac * len(present))))
        if args.max_scenes_per_bin > 0:
            k = min(k, args.max_scenes_per_bin)
        pick = rng.choice(np.array(present, dtype=object), size=min(k, len(present)), replace=False)
        nwin = 0
        for sid in pick:
            for wi in scene_wins[sid]:
                win_rows.append((wi, sid, bi, snr_of.get(sid, float("nan")))); nwin += 1
        print(f"  {lab}: {len(present)} scenes -> {len(pick)} sampled ({nwin} win)")

    N = len(win_rows)
    nchunks = int(np.ceil(N / args.chunk))
    print(f"TOTAL: {N} windows -> {nchunks} chunks of {args.chunk}")

    T = ds.window_samples
    targets = np.zeros((N, T), dtype=np.float32)
    sids = np.empty(N, dtype=object); binidx = np.zeros(N, dtype=np.int16)
    snrs = np.zeros(N, dtype=np.float32); chunk_id = np.zeros(N, dtype=np.int32); row = np.zeros(N, dtype=np.int32)
    ch, buf_a, buf_v, written = 0, [], [], 0
    for r, (wi, sid, b, sn) in enumerate(win_rows):
        item = ds[wi]
        targets[r] = item["target_audio"].numpy()[0]
        sids[r] = sid; binidx[r] = b; snrs[r] = sn
        buf_a.append(q_int16(item["mixed_audio"].numpy()[0], 15))
        buf_v.append(q_int16(item["video_frames"].numpy(), 9))
        chunk_id[r] = ch; row[r] = len(buf_a) - 1
        if len(buf_a) == args.chunk or r == N - 1:
            np.savez(outd / f"chunk_{ch:03d}_windows.npz",
                     audio_in=np.stack(buf_a), video_in=np.stack(buf_v))
            written += len(buf_a)
            print(f"  wrote chunk_{ch:03d}_windows.npz ({len(buf_a)} win, {written}/{N})", flush=True)
            ch += 1; buf_a, buf_v = [], []
    np.savez(outd / "meta.npz", target=targets, scene_id=np.array(sids), bin_idx=binidx, snr=snrs,
             chunk=chunk_id, row=row, nchunks=ch, chunk_size=args.chunk,
             bin_labels=np.array([b["label"] for b in bins]),
             bin_lo=np.array([b["low"] for b in bins]), bin_hi=np.array([b["high"] for b in bins]),
             full_bin_scene_counts=full_counts, frac=args.frac, seed=args.seed)
    print(f"DONE: {ch} chunks, {N} windows -> {outd}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
