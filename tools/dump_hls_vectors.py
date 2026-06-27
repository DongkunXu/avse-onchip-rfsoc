"""dump_hls_vectors.py — golden test vectors for the HLS C-sim cross-check (emulator == silicon).

For a few real dev windows, runs the fixed-point emulator (int16 + hardsigmoid) and dumps, per window:
  audio_in  [T]            = q_io(mixed)                       (sample_t the chip receives)
  video_embed [B*T_LAT]    = condition(video_encoder(video))  (data_t conditioning into audio_core)
  audio_out [T]            = emulator forward output           (the golden the HLS must reproduce)
Two vector files: `vectors_audio.txt` (audio_core: audio_in + video_embed -> audio_out) and
`vectors_full.txt` (monolithic: audio_in + raw video -> audio_out) for the B2 end-to-end check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))


def main() -> int:
    import argparse
    import contextlib
    import os
    from avse.config import Config
    from avse.data import AVSEDataset, AVSESceneStreamDataset
    from torch.utils.data import DataLoader
    from c7_fixedpoint import C7FixedPoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO / "experiments/p2-c7-full/deploy_weights.npz"))
    ap.add_argument("--nwin", type=int, default=2)
    ap.add_argument("--outdir", default=str(REPO / "hls/tb"))
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(str(REPO / "src/avse/config/onchip_config.yaml"))
    dev = "cpu"
    emu = C7FixedPoint(args.npz, device=dev, precision="int16", mask="hardsigmoid")

    with open(os.devnull, "w", encoding="utf-8") as dn, contextlib.redirect_stdout(dn):
        base = AVSEDataset(root_dir=cfg.data.root_dir, split="dev", config=cfg,
                           cache_dir=str(REPO / ".dataset_cache"))
    ds = AVSESceneStreamDataset(base, shuffle=False, max_windows=args.nwin)
    loader = DataLoader(ds, batch_size=args.nwin, num_workers=0)
    batch = next(iter(loader))
    mixed = batch["mixed_audio"].to(dev)              # [nw,1,T]
    video = batch["video_frames"].to(dev)             # [nw,TF,96,96]
    nw, _, T = mixed.shape

    with torch.no_grad():
        audio_in = emu.fx.io(mixed)                                   # sample_t input
        vfeat = emu._video(video)
        wlat = torch.nn.functional.conv1d(audio_in, emu.fx.wgt(emu.w["enc_w"]).unsqueeze(1),
                                          stride=emu.m["STRIDE"], padding=emu.m["STRIDE"])
        T_LAT = wlat.shape[-1]                                        # 1201
        vemb = emu._condition(vfeat, T_LAT)                          # [nw,B,T_LAT]
        out = emu.forward(mixed, video)                             # [nw,1,T]

    B = emu.m["B"]
    audio_in = audio_in.cpu().numpy()
    vemb = vemb.cpu().numpy()
    out = out.cpu().numpy()
    video_np = video.cpu().numpy()

    def w(f, arr):
        f.write(" ".join("%.8g" % v for v in np.asarray(arr).ravel()) + "\n")

    with open(Path(args.outdir) / "vectors_audio.txt", "w") as f:
        f.write(f"{nw} {T} {T_LAT} {B}\n")
        for i in range(nw):
            w(f, audio_in[i, 0])           # [T]
            w(f, vemb[i])                  # [B*T_LAT] row-major [b][t]
            w(f, out[i, 0])                # [T]
    print(f"vectors_audio.txt: {nw} windows, T={T} T_LAT={T_LAT} B={B}")

    with open(Path(args.outdir) / "vectors_full.txt", "w") as f:
        f.write(f"{nw} {T} {video_np.shape[1]} {video_np.shape[2]}\n")  # nw T TF IN
        for i in range(nw):
            w(f, audio_in[i, 0])           # [T]
            w(f, video_np[i])              # [TF*96*96]
            w(f, out[i, 0])                # [T]
    print(f"vectors_full.txt:  {nw} windows, TF={video_np.shape[1]} IN={video_np.shape[2]}")
    print(f"golden out rms (win0) = {np.sqrt((out[0]**2).mean()):.4e}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
