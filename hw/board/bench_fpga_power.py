"""bench_fpga_power.py — measure REAL FPGA inference power on the RFSoC 4x2 (board-side).

Reads the 9 on-board INA220 rails (/sys/class/hwmon/*/power1_input, microwatts) to get true board power.
Procedure: (1) load the static AVSE overlay; (2) sample rails for a few seconds IDLE (PL configured, IP not
running) -> idle baseline; (3) run a sustained inference loop (cycle the real windows N times) while sampling
the rails every window -> active power; report idle / active / dynamic, per-rail breakdown, mean compute
time, and energy per window. Uses REAL inputs (realistic switching activity, not zeros).

  sudo python3 bench_fpga_power.py --overlay avse_sys.bit --windows board_windows.npz --loops 30
"""
import argparse, glob, json, os, time
import numpy as np

AP_CTRL, OFF_AI, OFF_VI, OFF_AO = 0x00, 0x10, 0x1c, 0x28
T, TF, IN = 19200, 30, 96


def rail_handles():
    rails = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(os.path.join(h, "name")).read().strip() != "ina220":
                continue
            pf = os.path.join(h, "power1_input")
            if os.path.exists(pf):
                v = open(os.path.join(h, "in1_input")).read().strip() if os.path.exists(os.path.join(h, "in1_input")) else "?"
                rails.append((os.path.basename(h), pf, v))
        except Exception:
            pass
    return rails


def read_total_uw(rails):
    tot = 0; per = {}
    for name, pf, _ in rails:
        try:
            p = int(open(pf).read().strip()); tot += p; per[name] = p
        except Exception:
            per[name] = None
    return tot, per


def sample_idle(rails, secs):
    samples = []; t_end = time.time() + secs
    while time.time() < t_end:
        tot, _ = read_total_uw(rails); samples.append(tot); time.sleep(0.1)
    return np.array(samples) / 1e6      # W


def write64(mm, off, addr):
    mm.write(off, addr & 0xFFFFFFFF); mm.write(off + 4, (addr >> 32) & 0xFFFFFFFF)


def main():
    from pynq import Overlay, allocate
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="avse_sys.bit")
    ap.add_argument("--windows", default="board_windows.npz")
    ap.add_argument("--ip", default="avse_0")
    ap.add_argument("--loops", type=int, default=30, help="times to cycle through the window set")
    ap.add_argument("--idle-secs", type=float, default=4.0)
    ap.add_argument("--out", default="fpga_power.json")
    args = ap.parse_args()

    rails = rail_handles()
    print(f"INA220 rails: {[r[0] for r in rails]}")

    data = np.load(args.windows)
    ai = data["audio_in"]; vi = data["video_in"]
    nw = ai.shape[0]
    print(f"loaded {nw} real windows from {args.windows}; looping {args.loops}x = {nw*args.loops} inferences")

    ov = Overlay(args.overlay, download=True)
    ip = getattr(ov, args.ip); mm = ip.mmio
    b_ai = allocate(shape=(T,), dtype=np.int16, cacheable=False)
    b_vi = allocate(shape=(TF, IN, IN), dtype=np.int16, cacheable=False)
    b_ao = allocate(shape=(T,), dtype=np.int16, cacheable=False)
    write64(mm, OFF_AI, b_ai.physical_address); write64(mm, OFF_VI, b_vi.physical_address)
    write64(mm, OFF_AO, b_ao.physical_address)

    # idle baseline (overlay loaded, IP not started)
    print(f"sampling idle for {args.idle_secs}s ...", flush=True)
    idle = sample_idle(rails, args.idle_secs)
    print(f"  idle total board power = {idle.mean():.3f} W (min {idle.min():.3f}, max {idle.max():.3f})")

    # sustained inference loop, sampling rails every window
    psamples = []; per_accum = None; t_compute = []
    t0 = time.time()
    for L in range(args.loops):
        for i in range(nw):
            b_ai[:] = ai[i]; b_vi[:] = vi[i]; b_ai.flush(); b_vi.flush()
            ts = time.time(); mm.write(AP_CTRL, 1)
            while not (mm.read(AP_CTRL) & 0x2):
                pass
            t_compute.append(time.time() - ts)
            b_ao.invalidate()
            tot, per = read_total_uw(rails); psamples.append(tot)
            if per_accum is None:
                per_accum = {k: [] for k in per}
            for k, v in per.items():
                if v is not None: per_accum[k].append(v)
    wall = time.time() - t0
    act = np.array(psamples) / 1e6      # W
    cyc_ms = np.mean(t_compute) * 1e3
    per_rail_w = {k: float(np.mean(v)) / 1e6 for k, v in per_accum.items()}

    energy_j = act.mean() * (cyc_ms / 1e3)
    res = {
        "platform": "fpga", "device": "RFSoC 4x2 (xczu48dr) optimized bitstream", "precision": "int16",
        "clock_mhz": 187.5, "n_inferences": nw * args.loops,
        "latency_ms_per_window": cyc_ms,
        "throughput_win_per_s": 1.0 / np.mean(t_compute),
        "realtime_factor": (T / 16000.0) / (cyc_ms / 1e3),
        "audio_s_per_window": T / 16000.0,
        "idle_power_w": float(idle.mean()),
        "power_w": float(act.mean()),                 # active total board power
        "active_power_w_max": float(act.max()),
        "dynamic_power_w": float(act.mean() - idle.mean()),
        "per_rail_active_w": per_rail_w,
        "n_power_samples": len(act),
        "energy_j_per_window": energy_j,
        "energy_j_per_audio_s": energy_j / (T / 16000.0),
        "windows_per_joule": 1.0 / energy_j,
        "dynamic_energy_j_per_window": float(act.mean() - idle.mean()) * (cyc_ms / 1e3),
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n==== FPGA RFSoC 4x2 | int16 | {args.loops}x{nw} inferences ====")
    print(f"  latency/window : {cyc_ms:.1f} ms   (throughput {res['throughput_win_per_s']:.2f} win/s, "
          f"{res['realtime_factor']:.1f}x real-time)")
    print(f"  power idle     : {res['idle_power_w']:.2f} W")
    print(f"  power active   : {res['power_w']:.2f} W (dynamic +{res['dynamic_power_w']:.2f} W, "
          f"{res['n_power_samples']} samples)")
    print(f"  energy/window  : {energy_j*1e3:.0f} mJ total  ({res['dynamic_energy_j_per_window']*1e3:.0f} mJ dynamic)")
    print(f"  per-rail active (W): " + ", ".join(f"{k}={v:.2f}" for k, v in per_rail_w.items()))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
