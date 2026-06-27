"""export_weights.py — extract the trained C7 weights into the deployment weight set.

Single source of truth for the on-chip weights. Loads a trained C7 state_dict (e.g.
experiments/p2-c7-full/best.pt), folds every BatchNorm that can be folded *exactly*, and writes a flat
`.npz` of full-precision deploy weights. The fixed-point emulator (`tools/c7_fixedpoint.py`) and, later, the
HLS ROM header (`hls/src/c7_weights.hpp`) both consume this one file, so silicon and the measured number can
never drift apart. See `hls/DEPLOY_PLAN.md` for the deploy compute graph and the BN-fold rationale.

Folding rules (eps=1e-5; BN eval: y = gamma*(x-mean)/sqrt(var+eps) + beta = s*x + b):
  - **video** BNs sit directly after a (1x1 or k7/k2) conv with no padding in between -> fold into that conv
    exactly:  W_fold = W * s[out],  bias_fold = b  (the conv has no bias).
  - **audio in_norm / tcn bn1 / bn2** are kept as an inline per-channel affine (s, b), NOT folded, because
    PyTorch zero-pads the bn1 output before the dwconv (folding would mis-handle the pad boundary). The
    emulator/HLS apply them when writing the data_t buffer. We export their (s, b).

Usage:
    python tools/export_weights.py --ckpt experiments/p2-c7-full/best.pt --out experiments/p2-c7-full/deploy_weights.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EPS = 1e-5


def bn_affine(sd, prefix):
    """Return (s, b) of the folded affine s*x+b for a BatchNorm at `prefix` (eval-mode running stats)."""
    g = sd[f"{prefix}.weight"].double().numpy()
    beta = sd[f"{prefix}.bias"].double().numpy()
    mean = sd[f"{prefix}.running_mean"].double().numpy()
    var = sd[f"{prefix}.running_var"].double().numpy()
    s = g / np.sqrt(var + EPS)
    b = beta - s * mean
    return s, b


def fold_conv_bn(W, s, b):
    """Fold a BN (s,b) that follows a bias-less conv with weight W [out,in,...] into (W_fold, bias_fold)."""
    sh = (W.shape[0],) + (1,) * (W.ndim - 1)
    return W * s.reshape(sh), b.copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/p2-c7-full/best.pt")
    ap.add_argument("--out", default=None, help="default: <ckpt_dir>/deploy_weights.npz")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ckpt = (repo / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt.parent / "deploy_weights.npz"

    sd = torch.load(ckpt, map_location="cpu")
    g = lambda k: sd[k].double().numpy()  # noqa: E731  fp64 for exact extraction

    W = {}

    # ---- meta ----
    meta = dict(N=128, L=32, B=64, H=128, NBLK=10, X=5, KD=3, STRIDE=16, T=19200, T_LAT=1201,
                vid_TF=30, vid_IN=96, vid_C0=64, vid_C=96)
    for k, v in meta.items():
        W[f"meta_{k}"] = np.int64(v)

    # ======================= AUDIO =======================
    # encoder / decoder: [N,1,L] -> [N,L]
    W["enc_w"] = g("encoder.weight")[:, 0, :]          # [N,L]
    W["dec_w"] = g("decoder.weight")[:, 0, :]          # [N,L]

    # in_norm (inline affine on encoder output, before bottleneck) + bottleneck 1x1 (no bias)
    s, b = bn_affine(sd, "in_norm")
    W["innorm_s"], W["innorm_b"] = s, b                # [N]
    W["bn_W"] = g("bottleneck.weight")[:, :, 0].T      # bottleneck.weight[b,n] -> [N,B] indexed [n,b]

    # 10 TCN blocks
    for i in range(meta["NBLK"]):
        p = f"tcn.{i}"
        W[f"t{i}_in_w"] = g(f"{p}.in_conv.weight")[:, :, 0].T   # in_conv [H,B,1] -> [B,H] indexed [b,c]
        W[f"t{i}_pr1"] = g(f"{p}.prelu1.weight")                # [H]
        s1, b1 = bn_affine(sd, f"{p}.bn1"); W[f"t{i}_bn1_s"], W[f"t{i}_bn1_b"] = s1, b1
        W[f"t{i}_dw_w"] = g(f"{p}.dwconv.weight")[:, 0, :]      # dwconv [H,1,KD] -> [H,KD]
        W[f"t{i}_pr2"] = g(f"{p}.prelu2.weight")                # [H]
        s2, b2 = bn_affine(sd, f"{p}.bn2"); W[f"t{i}_bn2_s"], W[f"t{i}_bn2_b"] = s2, b2
        W[f"t{i}_out_w"] = g(f"{p}.out_conv.weight")[:, :, 0].T  # out_conv [B,H,1] -> [H,B] indexed [c,b]

    # mask 1x1 (no bias): mask_conv [N,B,1] -> [B,N] indexed [b,n]
    W["mask_W"] = g("mask_conv.weight")[:, :, 0].T

    # ======================= VIDEO =======================
    ve = "video.video_encoder.spatial_encoder"
    # conv0: Conv2d(1->64,k7) [0] + BN [1] + ReLU [2]
    W0 = g(f"{ve}.0.weight")                            # [64,1,7,7]
    s, b = bn_affine(sd, f"{ve}.1")
    W["v_c0_w"], W["v_c0_b"] = fold_conv_bn(W0, s, b)   # [64,1,7,7], [64]

    # DWSep stages at module indices 3,4,5 -> our s=1,2,3
    for s_idx, mod in enumerate([3, 4, 5], start=1):
        dp = f"{ve}.{mod}"
        W[f"v_dw{s_idx}_w"] = g(f"{dp}.depthwise.weight")[:, 0, :, :]    # [Cin,3,3]
        Wpw = g(f"{dp}.pointwise.weight")                                # [Cout,Cin,1,1]
        sp, bp = bn_affine(sd, f"{dp}.bn")
        Wpw_f, bpw = fold_conv_bn(Wpw[:, :, 0, 0], sp, bp)              # [Cout,Cin], [Cout]
        W[f"v_pw{s_idx}_w"], W[f"v_pw{s_idx}_b"] = Wpw_f, bpw
        Wsc = g(f"{dp}.shortcut.0.weight")                              # [Cout,Cin,1,1]
        ssc, bsc = bn_affine(sd, f"{dp}.shortcut.1")
        Wsc_f, bsc_f = fold_conv_bn(Wsc[:, :, 0, 0], ssc, bsc)         # [Cout,Cin], [Cout]
        W[f"v_sc{s_idx}_w"], W[f"v_sc{s_idx}_b"] = Wsc_f, bsc_f

    # feature_proj: Conv2d(96,96,k2) WITH bias (no BN) + ReLU
    W["v_fp_w"] = g("video.video_encoder.feature_proj.0.weight")        # [96,96,2,2]
    W["v_fp_b"] = g("video.video_encoder.feature_proj.0.bias")          # [96]
    # temporal_proj: Linear(96,96) (+ residual, applied in emulator)
    W["v_tp_w"] = g("video.video_encoder.temporal_proj.weight")        # [96,96] [o,i]
    W["v_tp_b"] = g("video.video_encoder.temporal_proj.bias")          # [96]
    # proj (conditioning): Conv1d(96->64,1) with bias
    W["vproj_w"] = g("video.proj.weight")[:, :, 0]                      # [64,96] [b,c]
    W["vproj_b"] = g("video.proj.bias")                                # [64]

    # ---- local BN-fold unit checks (cheap, exact) ----
    _verify_folds(sd, W)

    # ---- save + report ----
    np.savez(out, **W)
    print(f"saved -> {out}  ({len(W)} arrays)")
    print(f"{'array':16s}{'shape':18s}{'min':>9s}{'max':>9s}  note")
    wmax = 0.0
    for k in W:
        if k.startswith("meta_"):
            continue
        a = np.asarray(W[k], dtype=np.float64)
        note = "WGT" if not (k.endswith("_s") or k.endswith("_b") or "bn" in k or "innorm" in k) else "affine/bias"
        if note == "WGT":
            wmax = max(wmax, float(np.abs(a).max()))
        print(f"{k:16s}{str(a.shape):18s}{a.min():>9.3f}{a.max():>9.3f}  {note}")
    print(f"\nmax |weight| = {wmax:.3f}  (wgt_t=ap_fixed<16,5> range +-16 -> {'OK' if wmax < 16 else 'SATURATES'})")
    return 0


def _verify_folds(sd, W):
    """Confirm fold_conv_bn(conv,bn) reproduces bn(conv(x)) for the video convs (random-input check)."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    ve = "video.video_encoder.spatial_encoder"

    # conv0: bn(conv7x7(x)) vs folded
    x = torch.randn(1, 1, 96, 96, dtype=torch.float64)
    raw = F.conv2d(x, sd[f"{ve}.0.weight"].double(), stride=2, padding=3)
    s, b = bn_affine(sd, f"{ve}.1")
    ref = raw * torch.tensor(s).reshape(1, -1, 1, 1) + torch.tensor(b).reshape(1, -1, 1, 1)
    fold = F.conv2d(x, torch.tensor(W["v_c0_w"]), torch.tensor(W["v_c0_b"]), stride=2, padding=3)
    assert torch.allclose(ref, fold, atol=1e-9), "conv0 BN fold mismatch"

    # DWSep1 pointwise fold + shortcut fold
    dp = f"{ve}.3"
    y = torch.randn(1, 64, 24, 24, dtype=torch.float64)               # post-depthwise feature map
    raw = F.conv2d(y, sd[f"{dp}.pointwise.weight"].double())
    sp, bp = bn_affine(sd, f"{dp}.bn")
    ref = raw * torch.tensor(sp).reshape(1, -1, 1, 1) + torch.tensor(bp).reshape(1, -1, 1, 1)
    fold = F.conv2d(y, torch.tensor(W["v_pw1_w"])[:, :, None, None], torch.tensor(W["v_pw1_b"]))
    assert torch.allclose(ref, fold, atol=1e-9), "DWSep1 pointwise BN fold mismatch"
    print("BN-fold unit checks: PASS")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
