"""score_board.py — PC-side: score the FPGA's int16 outputs and confirm silicon == emulator.

Loads the board run's outputs + the held-back targets, computes SI-SDR / PESQ / STOI (scene-weighted, the
eval_deploy protocol), and compares to the C-sim-validated emulator number (4.984 / 1.632 / 0.742 full dev).
With --compare-emulator it also feeds the *same* int16 inputs through the fixed-point emulator and reports
the FPGA-vs-emulator output difference — the final on-silicon confirmation that the chip computes what we
measured.

    python tools/score_board.py --outputs hw/board/board_outputs.npz --targets hw/board/board_targets.npz
    python tools/score_board.py ... --compare-emulator --windows hw/board/board_windows.npz
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from avse.metrics import si_sdr, pesq_wb, stoi_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(REPO / "hw/board/board_outputs.npz"))
    ap.add_argument("--targets", default=str(REPO / "hw/board/board_targets.npz"))
    ap.add_argument("--windows", default=str(REPO / "hw/board/board_windows.npz"))
    ap.add_argument("--compare-emulator", action="store_true")
    args = ap.parse_args()

    out = np.load(args.outputs)["audio_out"].astype(np.float64) / 32768.0   # sample_t -> float
    tgt_npz = np.load(args.targets, allow_pickle=True)
    target, mixed = tgt_npz["target"], tgt_npz["mixed"]
    sids = tgt_npz["scene_id"] if "scene_id" in tgt_npz.files else np.arange(len(out)).astype(str)
    N = out.shape[0]
    print(f"scoring {N} FPGA windows")

    acc = defaultdict(lambda: defaultdict(list))
    for i in range(N):
        e, t, m = out[i], target[i], mixed[i]
        sid = str(sids[i])
        acc[sid]["si_e"].append(si_sdr(e, t)); acc[sid]["si_m"].append(si_sdr(m, t))
        acc[sid]["p_e"].append(pesq_wb(t, e)); acc[sid]["p_m"].append(pesq_wb(t, m))
        acc[sid]["s_e"].append(stoi_score(t, e)); acc[sid]["s_m"].append(stoi_score(t, m))

    def overall(k):
        vals = [np.mean(a[k]) for a in acc.values() if a[k]]
        return float(np.mean(vals)) if vals else None

    print("\n========  ON-BOARD (FPGA) RESULT  ========")
    print(f"scenes={len(acc)} windows={N}")
    print(f"{'':10s}{'SI-SDR':>10s}{'PESQ':>9s}{'STOI':>8s}")
    print(f"{'enhanced':10s}{overall('si_e'):>10.3f}{overall('p_e'):>9.3f}{overall('s_e'):>8.3f}")
    print(f"{'mixed':10s}{overall('si_m'):>10.3f}{overall('p_m'):>9.3f}{overall('s_m'):>8.3f}")
    print("ref (emulator, full dev): SI-SDR 4.984  PESQ 1.632  STOI 0.742")

    if args.compare_emulator:
        import torch
        from c7_fixedpoint import C7FixedPoint
        win = np.load(args.windows)
        ai = win["audio_in"].astype(np.float32) / 32768.0       # back to float (exact, on-grid)
        vi = win["video_in"].astype(np.float32) / 512.0
        emu = C7FixedPoint(REPO / "experiments/p2-c7-full/deploy_weights.npz",
                           device="cpu", precision="int16", mask="hardsigmoid")
        with torch.no_grad():
            est = emu.forward(torch.tensor(ai)[:, None, :], torch.tensor(vi)).numpy()[:, 0, :]
        d = np.abs(out - est)
        print(f"\n[silicon vs emulator] max|diff|={d.max():.3e}  rms_diff={np.sqrt((d**2).mean()):.3e}  "
              f"(out rms={np.sqrt((out**2).mean()):.3e}) -> {'MATCH' if d.max() < 5e-3 else 'CHECK'}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
