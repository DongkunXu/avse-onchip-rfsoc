"""prep_board_novideo_chunks.py — build AUDIO-ONLY board chunks for the on-chip visual ablation.

The on-board visual ablation feeds a zeroed video buffer (run_fpga.py --zero-video), so the per-chunk
upload doesn't need the video at all. This reads the existing SNR-bin chunks
(hw/board/snr_eval/chunk_*_windows.npz, which carry the REAL int16 audio_in + video_in) and writes
audio-only chunks (same audio_in, no video_in) compressed — a few MB each instead of ~295 MB, so the
upload is fast. The audio is byte-identical to the with-video run, so the ONLY difference on the board
is the zeroed video → a clean A/B on silicon.

  python tools/prep_board_novideo_chunks.py --src hw/board/snr_eval --dst hw/board/snr_eval_novideo
"""
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "hw/board/snr_eval"))
    ap.add_argument("--dst", default=str(REPO / "hw/board/snr_eval_novideo"))
    args = ap.parse_args()
    src = Path(args.src); dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    chunks = sorted(src.glob("chunk_*_windows.npz"))
    if not chunks:
        print(f"no chunk_*_windows.npz in {src}"); return 1
    print(f"{len(chunks)} chunks: {src} -> {dst} (audio-only, compressed)")
    for w in chunks:
        a = np.load(w)["audio_in"]            # [n,19200] int16 — same audio as the with-video run
        out = dst / w.name
        np.savez_compressed(out, audio_in=a)
        print(f"  {w.name}: audio_in {a.shape} {a.dtype} -> {out.stat().st_size/1e6:.1f} MB", flush=True)
    print(f"DONE -> {dst}  (run on board with run_fpga.py --zero-video)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
