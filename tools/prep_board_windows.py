"""prep_board_windows.py — PC-side: quantize dev windows to int16 for the on-board FPGA run.

Produces the exact int16 raw bits the IP expects (matching the fixed-point emulator that measured 4.98 dB),
plus the targets/mixed for scoring. The board runs at ~2.5 s/window (rolled video), so a subset confirms the
silicon reproduces the validated output (the full-dev quality is already established by the C-sim-validated
emulator). Outputs board_windows.npz (-> copy to board) and board_targets.npz (stays on PC for scoring).

    python tools/prep_board_windows.py --nwin 200 --out-dir hw/board
"""
import argparse
import os
import sys
import contextlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def q_int16(x, frac, lo=-32768, hi=32767):
    """Raw int16 of ap_fixed with `frac` fractional bits, AP_TRN (floor) + AP_SAT — matches the emulator."""
    q = np.floor(np.asarray(x, dtype=np.float64) * (2 ** frac))
    return np.clip(q, lo, hi).astype(np.int16)


def main():
    import torch
    from torch.utils.data import DataLoader
    from avse.config import Config
    from avse.data import AVSEDataset, AVSESceneStreamDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--nwin", type=int, default=200)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out-dir", default=str(REPO / "hw/board"))
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        base = AVSEDataset(root_dir=cfg.data.root_dir, split=args.split, config=cfg,
                           cache_dir=str(REPO / ".dataset_cache"))
    ds = AVSESceneStreamDataset(base, shuffle=False, max_windows=args.nwin)
    loader = DataLoader(ds, batch_size=args.nwin, num_workers=0)
    batch = next(iter(loader))

    mixed = batch["mixed_audio"].numpy()[:, 0, :]     # [N,19200]
    video = batch["video_frames"].numpy()             # [N,30,96,96] in [0,1]
    target = batch["target_audio"].numpy()[:, 0, :]   # [N,19200]
    sids = list(batch["scene_id"])
    N = mixed.shape[0]

    audio_in = q_int16(mixed, 15)                     # sample_t <16,1>
    video_in = q_int16(video, 9)                      # data_t   <16,7>

    np.savez(Path(args.out_dir) / "board_windows.npz", audio_in=audio_in, video_in=video_in)
    np.savez(Path(args.out_dir) / "board_targets.npz", target=target, mixed=mixed,
             scene_id=np.array(sids))
    print(f"prepped {N} windows -> {args.out_dir}/board_windows.npz "
          f"(audio_in {audio_in.shape} {audio_in.dtype}, video_in {video_in.shape})")
    print(f"  audio_in range [{audio_in.min()},{audio_in.max()}], video_in range [{video_in.min()},{video_in.max()}]")
    print(f"  targets/mixed -> {args.out_dir}/board_targets.npz (scoring stays on PC)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
