"""run_fpga.py — drive the single-config C7 AVSE bitstream on the RFSoC 4x2 (PYNQ), board-side.

Runs ON THE BOARD. Loads the static overlay, streams pre-quantized int16 windows through the one IP
(c7_avse_top), and saves the int16 enhanced output. The whole AVSE (audio + video) runs in ONE static
bitstream — no PCAP reconfig (the project's thesis).

Pipeline split (avoids needing torch / the dataset on the board):
  PC   : tools/prep_board_windows.py  -> board_windows.npz  (int16 audio_in [N,19200], video_in [N,30,96,96])
  BOARD: this script                  -> board_outputs.npz  (int16 audio_out [N,19200])
  PC   : tools/score_board.py         -> SI-SDR/PESQ/STOI vs targets, compared to the 4.98 dB emulator

int16 raw bits are the fixed-point values the IP expects/produces, matching the emulator exactly:
  audio_in  sample_t ap_fixed<16,1>  -> int16 = clip(floor(x*2^15), -32768, 32767)   (set by prep)
  video_in  data_t   ap_fixed<16,7>  -> int16 = clip(floor(x*2^9),  -32768, 32767)   (set by prep)
  audio_out sample_t ap_fixed<16,1>  -> float = int16 / 2^15                          (here)

s_axilite register map (from xc7_avse_top_hw.h): 0x00 AP_CTRL, 0x10 audio_in*, 0x1c video_in*, 0x28 audio_out*.

Usage (on the board):  sudo python3 run_fpga.py --overlay avse_sys.bit --windows board_windows.npz --out board_outputs.npz
"""
import argparse
import time
import numpy as np

AP_CTRL = 0x00
OFF_AUDIO_IN = 0x10
OFF_VIDEO_IN = 0x1c
OFF_AUDIO_OUT = 0x28
T = 19200
TF, IN = 30, 96


def write64(mm, off, addr):
    mm.write(off, addr & 0xFFFFFFFF)
    mm.write(off + 4, (addr >> 32) & 0xFFFFFFFF)


def main():
    from pynq import Overlay, allocate

    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="avse_sys.bit")
    ap.add_argument("--windows", default="board_windows.npz")
    ap.add_argument("--out", default="board_outputs.npz")
    ap.add_argument("--ip", default="avse_0", help="IP instance name in the overlay")
    args = ap.parse_args()

    data = np.load(args.windows)
    audio_in_all = data["audio_in"]      # [N,19200] int16
    video_in_all = data["video_in"]      # [N,30,96,96] int16
    N = audio_in_all.shape[0]
    print(f"loaded {N} windows from {args.windows}")

    ov = Overlay(args.overlay, download=True)
    ip = getattr(ov, args.ip)
    mm = ip.mmio

    # one set of DDR buffers, reused across windows
    b_audio_in = allocate(shape=(T,), dtype=np.int16, cacheable=False)
    b_video_in = allocate(shape=(TF, IN, IN), dtype=np.int16, cacheable=False)
    b_audio_out = allocate(shape=(T,), dtype=np.int16, cacheable=False)
    write64(mm, OFF_AUDIO_IN, b_audio_in.physical_address)
    write64(mm, OFF_VIDEO_IN, b_video_in.physical_address)
    write64(mm, OFF_AUDIO_OUT, b_audio_out.physical_address)

    outputs = np.zeros((N, T), dtype=np.int16)
    t_compute = []
    for i in range(N):
        b_audio_in[:] = audio_in_all[i]
        b_video_in[:] = video_in_all[i]
        b_audio_in.flush(); b_video_in.flush()

        t0 = time.time()
        mm.write(AP_CTRL, 1)                       # ap_start
        while not (mm.read(AP_CTRL) & 0x2):        # poll ap_done (bit1)
            pass
        t_compute.append(time.time() - t0)

        b_audio_out.invalidate()
        outputs[i] = np.array(b_audio_out)
        print(f"  {i+1}/{N} | compute {t_compute[-1]:.2f}s | out[min,max]=[{outputs[i].min()},{outputs[i].max()}]",
              flush=True)

    np.savez(args.out, audio_out=outputs)
    print(f"saved {args.out} | mean compute = {np.mean(t_compute)*1e3:.1f} ms/window "
          f"({1.0/np.mean(t_compute):.2f} win/s)")
    print(f"out int16 range = [{outputs.min()}, {outputs.max()}]")


if __name__ == "__main__":
    main()
