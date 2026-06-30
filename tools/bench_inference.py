"""bench_inference.py — latency / throughput / power / energy of the C7 AVSE forward on CPU or GPU.

Runs the FP32 model (experiments/p2-c7-full/best.pt) — the as-deployed software path — on one 1.2 s window
(19200 samples) at a time (batch=1, matching the FPGA's streaming mode) or batched (GPU max throughput).
Times a warmed-up sustained loop with proper CUDA sync; on GPU samples real board power via NVML
(pynvml) or `nvidia-smi` in a background thread. Inputs are random of the correct shape (power/latency are
value-independent). Saves a JSON row for the cross-platform table built by summarize_efficiency.py.

  python tools/bench_inference.py --device cuda --batch 1
  python tools/bench_inference.py --device cuda --batch 64
  python tools/bench_inference.py --device cpu  --batch 1
"""
import argparse, json, os, sys, time, threading, subprocess
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SR = 16000
WIN = 19200                       # samples/window
AUDIO_S = WIN / SR                # 1.2 s of audio per window


class GpuPowerSampler:
    """Poll GPU power (W) and SM clock in a background thread. Prefers NVML, falls back to nvidia-smi."""
    def __init__(self, period=0.05):
        self.period = period; self.samples = []; self.clocks = []; self._stop = False
        self._nvml = None; self._h = None
        try:
            import pynvml
            pynvml.nvmlInit(); self._nvml = pynvml; self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _read(self):
        if self._nvml:
            p = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            try: c = self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_SM)
            except Exception: c = 0
            return p, c
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        a = out.stdout.strip().splitlines()[0].split(",")
        return float(a[0]), float(a[1])

    def _loop(self):
        while not self._stop:
            try:
                p, c = self._read(); self.samples.append(p); self.clocks.append(c)
            except Exception:
                pass
            time.sleep(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self

    def __exit__(self, *a):
        self._stop = True; self._t.join(timeout=2)

    def stats(self):
        if not self.samples:
            return None
        s = np.array(self.samples)
        return {"power_w_mean": float(s.mean()), "power_w_max": float(s.max()),
                "power_w_p50": float(np.median(s)), "n_samples": len(s),
                "sm_clock_mhz_mean": float(np.mean(self.clocks)) if self.clocks else None}


def main():
    import torch
    from avse.config import Config
    from avse.models import ConvTasNetAVSE

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0, help="cpu torch threads (0=default/all)")
    ap.add_argument("--ckpt", default=str(REPO / "experiments/p2-c7-full/best.pt"))
    ap.add_argument("--config", default=str(REPO / "src/avse/config/onchip_config.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "experiments/power_efficiency"))
    args = ap.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg = Config.from_yaml(args.config)
    model = ConvTasNetAVSE(cfg).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))

    torch.manual_seed(0)
    vid = torch.randn(args.batch, 30, 96, 96, device=dev)
    aud = torch.randn(args.batch, 1, WIN, device=dev)
    batch = {"video_frames": vid, "mixed_audio": aud}

    def run_n(n):
        with torch.no_grad():
            for _ in range(n):
                model(batch)

    # warmup
    run_n(args.warmup)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # timed loop (+ GPU power sampling)
    sampler = GpuPowerSampler() if dev.type == "cuda" else None
    cpu_info = None
    if dev.type == "cuda":
        with sampler:
            t0 = time.perf_counter(); run_n(args.iters); torch.cuda.synchronize(); t1 = time.perf_counter()
        pstats = sampler.stats()
    else:
        try:
            import platform
            cpu_info = platform.processor()
        except Exception:
            pass
        t0 = time.perf_counter(); run_n(args.iters); t1 = time.perf_counter()
        pstats = None

    total_s = t1 - t0
    n_windows = args.iters * args.batch
    lat_ms = total_s / n_windows * 1000.0          # per single window
    thr = n_windows / total_s                      # windows / s
    rt_factor = AUDIO_S / (lat_ms / 1000.0)         # x real-time (1.2 s audio per window)

    res = {
        "platform": f"{args.device}",
        "device_name": (torch.cuda.get_device_name(0) if dev.type == "cuda" else (cpu_info or "cpu")),
        "precision": "fp32",
        "batch": args.batch, "iters": args.iters, "warmup": args.warmup,
        "cpu_threads": (torch.get_num_threads() if dev.type == "cpu" else None),
        "latency_ms_per_window": lat_ms,
        "throughput_win_per_s": thr,
        "realtime_factor": rt_factor,
        "audio_s_per_window": AUDIO_S,
        "power_w": (pstats["power_w_mean"] if pstats else None),
        "power_stats": pstats,
        "energy_j_per_window": (pstats["power_w_mean"] * (lat_ms / 1000.0) if pstats else None),
    }
    if res["energy_j_per_window"] is not None:
        res["energy_j_per_audio_s"] = res["energy_j_per_window"] / AUDIO_S
        res["windows_per_joule"] = 1.0 / res["energy_j_per_window"]

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    tag = f"{args.device}_b{args.batch}"
    json.dump(res, open(outd / f"bench_{tag}.json", "w", encoding="utf-8"), indent=2)

    print(f"\n==== {res['device_name']} | {args.device} | batch={args.batch} | fp32 ====")
    print(f"  latency/window : {lat_ms:.3f} ms   (throughput {thr:.1f} win/s, {rt_factor:.1f}x real-time)")
    if pstats:
        print(f"  power (active) : {pstats['power_w_mean']:.1f} W mean / {pstats['power_w_max']:.1f} W max "
              f"(SM {pstats['sm_clock_mhz_mean']:.0f} MHz, {pstats['n_samples']} samples)")
        print(f"  energy/window  : {res['energy_j_per_window']*1000:.1f} mJ   "
              f"({res['energy_j_per_audio_s']*1000:.1f} mJ per audio-second, {res['windows_per_joule']:.1f} win/J)")
    else:
        print(f"  power          : NOT measured on CPU (estimate applied in summary)")
    print(f"  saved -> {outd / f'bench_{tag}.json'}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
