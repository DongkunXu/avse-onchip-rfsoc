"""verify_data.py — end-to-end check that the migrated LRS3 pipeline loads on THIS machine.

De-risks all of Phase 2 before any training: builds the dataset on a split, loads a sample and a
batch, and reports shapes + value ranges. Run:

    .venv/Scripts/python.exe tools/verify_data.py --split dev --max-scenes 40
"""
from __future__ import annotations

import sys, argparse, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from pathlib import Path
import torch
from torch.utils.data import DataLoader

# make `avse` importable without install side-effects (it's pip -e installed, but be robust)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avse.config import Config
from avse.data import AVSEDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = use all scenes in the split")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"root_dir = {cfg.data.root_dir} | split = {args.split}")
    print(f"window = {cfg.data.window_duration}s @ {cfg.audio.sample_rate}Hz | video {cfg.video.input_size} @ {cfg.video.fps}fps")

    t0 = time.time()
    ds = AVSEDataset(root_dir=cfg.data.root_dir, split=args.split, config=cfg)
    print(f"\nDataset built in {time.time()-t0:.1f}s — {len(ds)} windows")
    if len(ds) == 0:
        print("!! 0 windows — check the dataset layout / paths."); return 1

    # sample 0
    s = ds[0]
    v, mx, tg = s["video_frames"], s["mixed_audio"], s["target_audio"]
    print("\nsample[0]:")
    print(f"  video_frames {tuple(v.shape)} {v.dtype}  range[{v.min():.3f},{v.max():.3f}]")
    print(f"  mixed_audio  {tuple(mx.shape)} {mx.dtype}  range[{mx.min():.3f},{mx.max():.3f}]")
    print(f"  target_audio {tuple(tg.shape)} {tg.dtype}  range[{tg.min():.3f},{tg.max():.3f}]")
    print(f"  scene_id={s['scene_id']}  window_start={s['window_start']}")

    # one batch
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    b = next(iter(dl))
    print(f"\nbatch (bs={args.batch}): video {tuple(b['video_frames'].shape)}  "
          f"mixed {tuple(b['mixed_audio'].shape)}  target {tuple(b['target_audio'].shape)}")

    # expected shapes from config
    exp_T = int(cfg.data.window_duration * cfg.audio.sample_rate)
    exp_F = int(cfg.data.window_duration * cfg.video.fps)
    print(f"\nexpected: audio T={exp_T}, video frames={exp_F}, frame {cfg.video.input_size}")
    okT = b["mixed_audio"].shape[-1] == exp_T
    okF = b["video_frames"].shape[1] == exp_F
    print(f"audio length match: {okT} | video frame-count match: {okF}")

    print("\nDATA PIPELINE OK" if (okT and okF) else "\nDATA PIPELINE shapes differ — inspect above")
    return 0 if (okT and okF) else 1


if __name__ == "__main__":
    sys.exit(main())
