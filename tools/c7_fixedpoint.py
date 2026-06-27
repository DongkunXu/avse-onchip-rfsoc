"""c7_fixedpoint.py — bit-faithful fixed-point emulator of the C7 on-chip AVSE deployment.

Re-implements the *deployed* compute graph (video encoder + audio mask network) exactly as the HLS design
computes it (`hls/DEPLOY_PLAN.md`): the same op order, the same buffer-write quantization points, folded /
inline BN, hardsigmoid mask, and the corrected encoder/decoder indexing (G1 T_LAT=1201, G2 decoder offset).
Weights come from the one exported set (`deploy_weights.npz`), so this emulator and the HLS ROMs are the same
model. Runs batched on GPU.

Fixed-point policy (DECISIONS D-18):
  - I/O PCM            -> sample_t  ap_fixed<16,1>  (frac 15)
  - activations (buffers) -> data_t ap_fixed<16,7>  (frac 9)   <- the real quality limiter, quantized
  - MAC weight operands   -> wgt_t  ap_fixed<16,5>  (frac 11)  quantized
  - inline BN/in_norm affine (s,b) and PReLU slopes -> kept high precision (constants, not datapath operands;
    the int16 lock, D-3, is on the BRAM-dominating activation/weight datapath). This also avoids the
    in_norm-scale (up to ~102) overflowing wgt_t.
  - accumulators wide (acc_t) -> treated as exact.

`precision="fp"` disables all quantizers -> must reproduce the PyTorch best.pt forward (the correctness gate).
`precision="int16"` applies the policy above -> the deployment-accurate forward.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------- fixed-point quantizers -----------------------------
def _q(x, frac, intb, sat=True):
    """ap_fixed<frac+intb, intb> with AP_TRN (floor toward -inf) and optional AP_SAT."""
    step = 2.0 ** (-frac)
    q = torch.floor(x / step) * step
    if sat:
        lo = -(2.0 ** (intb - 1))
        hi = 2.0 ** (intb - 1) - step
        q = torch.clamp(q, lo, hi)
    return q


class FX:
    """Quantizer bundle; identity when enabled=False (fp mode)."""
    def __init__(self, enabled: bool):
        self.on = enabled

    def act(self, x):   # data_t  <16,7>
        return _q(x, 9, 7) if self.on else x

    def wgt(self, x):   # wgt_t   <16,5>
        return _q(x, 11, 5) if self.on else x

    def io(self, x):    # sample_t <16,1>
        return _q(x, 15, 1) if self.on else x


# ----------------------------- the emulator -----------------------------
class C7FixedPoint:
    """Faithful fixed-point emulator built from deploy_weights.npz. Call .forward(mixed, video)."""

    def __init__(self, npz_path, device="cpu", precision="int16", mask="hardsigmoid"):
        z = np.load(npz_path)
        self.dev = device
        self.fx = FX(precision == "int16")
        assert mask in ("hardsigmoid", "sigmoid")
        self.mask_kind = mask  # hardsigmoid = the HW deployment; sigmoid = the trained reference
        self.m = {k[5:]: int(z[k]) for k in z.files if k.startswith("meta_")}
        t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=device)  # noqa: E731
        self.w = {k: t(k) for k in z.files if not k.startswith("meta_")}

    # ---- video encoder: video[B,TF,96,96] -> video_feat[B,C,TF] ----
    def _video(self, vid):
        fx, w, m = self.fx, self.w, self.m
        Bn, TF, Hh, Ww = vid.shape
        C0, C = m["vid_C0"], m["vid_C"]
        x = fx.act(vid.reshape(Bn * TF, 1, Hh, Ww))                       # frames as data_t

        # conv0: Conv2d(1->64,k7,s2,p3) + BN(folded) + ReLU
        x = F.conv2d(x, fx.wgt(w["v_c0_w"]), w["v_c0_b"], stride=2, padding=3)
        x = fx.act(F.relu(x))                                            # b0 [*,64,48,48]

        def dwsep(x, s_idx, cin, cout):
            dw = F.conv2d(x, fx.wgt(w[f"v_dw{s_idx}_w"]).unsqueeze(1), stride=2, padding=1, groups=cin)
            dw = fx.act(dw)                                              # depthwise (no activation in PyTorch)
            pw = F.conv2d(dw, fx.wgt(w[f"v_pw{s_idx}_w"])[:, :, None, None], w[f"v_pw{s_idx}_b"])
            main = fx.act(F.relu(pw))                                    # bn+relu (main branch)
            sc = F.conv2d(x, fx.wgt(w[f"v_sc{s_idx}_w"])[:, :, None, None], w[f"v_sc{s_idx}_b"], stride=2)
            return fx.act(main + fx.act(sc))                            # residual add

        x = dwsep(x, 1, C0, C)                                           # [*,96,24,24]
        x = dwsep(x, 2, C, C)                                            # [*,96,12,12]
        x = dwsep(x, 3, C, C)                                            # [*,96,6,6]

        x = fx.act(F.avg_pool2d(x, kernel_size=5, stride=1))            # 6x6 -> 2x2
        x = F.conv2d(x, fx.wgt(w["v_fp_w"]), w["v_fp_b"])               # feature_proj k2 -> 1x1
        x = fx.act(F.relu(x)).reshape(Bn * TF, C)                       # [*,96]

        # temporal_proj Linear(96,96) + residual
        tp = F.linear(x, fx.wgt(w["v_tp_w"]), w["v_tp_b"])
        x = fx.act(tp + x)
        return x.reshape(Bn, TF, C).transpose(1, 2)                    # [B,C,TF]

    # ---- conditioning: video_feat[B,C,TF] -> video_embed[B,B_ch,T_LAT] ----
    def _condition(self, vfeat, t_lat):
        fx, w = self.fx, self.w
        v = F.conv1d(vfeat, fx.wgt(w["vproj_w"]).unsqueeze(-1), w["vproj_b"])   # proj 96->64
        v = fx.act(v)
        return F.interpolate(v, size=t_lat, mode="nearest")                     # 30 -> T_LAT (nearest)

    # ---- audio mask network ----
    def forward(self, mixed, video):
        fx, w, m = self.fx, self.w, self.m
        N, Bc, H, KD, STR = m["N"], m["B"], m["H"], m["KD"], m["STRIDE"]
        x = fx.io(mixed)                                                 # [B,1,T] PCM

        # encoder Conv1d(1->N,k=L,s=STR,pad=STR)
        wlat = fx.act(F.conv1d(x, fx.wgt(w["enc_w"]).unsqueeze(1), stride=STR, padding=STR))  # [B,N,T_LAT]
        T_LAT = wlat.shape[-1]

        vfeat = self._video(video)
        vemb = self._condition(vfeat, T_LAT)                            # [B,Bc,T_LAT]

        # bottleneck: y = (in_norm(w) -> 1x1) + video   (in_norm inline, wn cast to data_t)
        wn = fx.act(w["innorm_s"][None, :, None] * wlat + w["innorm_b"][None, :, None])
        y = F.conv1d(wn, fx.wgt(w["bn_W"].T).unsqueeze(-1))             # bn_W[n,b]->weight[b,n,1]
        y = fx.act(y + vemb)

        # 10 dilated dwsep TCN blocks (residual)
        for i in range(m["NBLK"]):
            dil = 1 << (i % m["X"])
            # IN1x1 -> prelu1 -> bn1   (h, data_t)
            a = F.conv1d(y, fx.wgt(w[f"t{i}_in_w"].T).unsqueeze(-1))    # in_w[b,c]->weight[c,b,1]
            h = F.prelu(a, w[f"t{i}_pr1"])
            h = fx.act(w[f"t{i}_bn1_s"][None, :, None] * h + w[f"t{i}_bn1_b"][None, :, None])
            # depthwise (causal: left pad (KD-1)*dil) -> prelu2 -> bn2   (hd, data_t)
            hp = F.pad(h, ((KD - 1) * dil, 0))
            dwk = fx.wgt(w[f"t{i}_dw_w"]).unsqueeze(1)                  # [H,1,KD]
            hd = F.conv1d(hp, dwk, dilation=dil, groups=H)
            hd = F.prelu(hd, w[f"t{i}_pr2"])
            hd = fx.act(w[f"t{i}_bn2_s"][None, :, None] * hd + w[f"t{i}_bn2_b"][None, :, None])
            # OUT1x1 -> residual add   (y, data_t)
            o = F.conv1d(hd, fx.wgt(w[f"t{i}_out_w"].T).unsqueeze(-1))  # out_w[c,b]->weight[b,c,1]
            y = fx.act(y + o)

        # mask: mask_conv(y) -> hardsigmoid (HW) or sigmoid (trained ref); w *= mask
        a = F.conv1d(y, fx.wgt(w["mask_W"].T).unsqueeze(-1))           # mask_W[b,n]->weight[n,b,1]
        a = torch.clamp(0.2 * a + 0.5, 0.0, 1.0) if self.mask_kind == "hardsigmoid" else torch.sigmoid(a)
        mask = fx.act(a)                                               # -> data_t
        wmasked = fx.act(wlat * mask)

        # decoder ConvTranspose1d(N->1,k=L,s=STR,pad=STR)  (correct offset; G2)
        out = F.conv_transpose1d(wmasked, fx.wgt(w["dec_w"]).unsqueeze(1), stride=STR, padding=STR)
        out = self.fx.io(out)
        T = m["T"]
        return out[..., :T] if out.shape[-1] >= T else F.pad(out, (0, T - out.shape[-1]))


# ----------------------------- self-test / correctness gate -----------------------------
def _selftest():
    """fp-mode emulator must reproduce the PyTorch best.pt forward (proves the reimplementation + fold)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from avse.config import Config
    from avse.models import ConvTasNetAVSE

    repo = Path(__file__).resolve().parents[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config.from_yaml(str(repo / "src/avse/config/onchip_config.yaml"))
    ref = ConvTasNetAVSE(cfg).to(dev).eval()
    ref.load_state_dict(torch.load(repo / "experiments/p2-c7-full/best.pt", map_location=dev))

    npz = repo / "experiments/p2-c7-full/deploy_weights.npz"
    # correctness gate uses sigmoid (the trained nonlinearity); hardsigmoid is the deployment choice
    emu_fp = C7FixedPoint(npz, device=dev, precision="fp", mask="sigmoid")
    emu_q = C7FixedPoint(npz, device=dev, precision="int16", mask="hardsigmoid")
    emu_q_sig = C7FixedPoint(npz, device=dev, precision="int16", mask="sigmoid")

    torch.manual_seed(0)
    Bn, T, TF = 2, 19200, 30
    mixed = torch.randn(Bn, 1, T, device=dev) * 0.1
    video = torch.rand(Bn, TF, 96, 96, device=dev)

    with torch.no_grad():
        y_ref = ref({"mixed_audio": mixed, "video_frames": video})
        y_fp = emu_fp.forward(mixed, video)
        y_q = emu_q.forward(mixed, video)
        y_qs = emu_q_sig.forward(mixed, video)

    d_fp = (y_ref - y_fp).abs().max().item()
    rms = y_ref.pow(2).mean().sqrt().item()
    print(f"[gate] emulator-fp (sigmoid) vs best.pt : max|diff|={d_fp:.3e}  (ref rms={rms:.3e})")
    print(f"[info] int16(sigmoid)   vs best.pt: max|diff|={(y_ref - y_qs).abs().max().item():.3e}  (pure quant cost)")
    print(f"[info] int16(hardsig)   vs best.pt: max|diff|={(y_ref - y_q).abs().max().item():.3e}  (quant + hardsigmoid)")
    ok = d_fp < 1e-3
    print("GATE:", "PASS" if ok else "FAIL (reimplementation/fold differs from PyTorch)")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(_selftest())
