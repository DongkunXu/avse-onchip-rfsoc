"""diag.py — board-side overlay/IP diagnostic (run BEFORE the full run_fpga to isolate issues)."""
import time
import numpy as np
from pynq import Overlay, allocate

print("loading overlay...", flush=True)
t = time.time()
ov = Overlay("avse_sys.bit", download=True)
print(f"overlay loaded in {time.time()-t:.1f}s", flush=True)
print("ips:", list(ov.ip_dict.keys()), flush=True)

ip = ov.avse_0
print("register_map:", flush=True)
print(ip.register_map, flush=True)

# one tiny run: zero buffers, start, poll ap_done with a timeout
mm = ip.mmio
T = 19200
ai = allocate(shape=(T,), dtype=np.int16, cacheable=False)
vi = allocate(shape=(30, 96, 96), dtype=np.int16, cacheable=False)
ao = allocate(shape=(T,), dtype=np.int16, cacheable=False)
ai[:] = 0; vi[:] = 0; ao[:] = 0
ai.flush(); vi.flush(); ao.flush()


def w64(off, a):
    mm.write(off, a & 0xFFFFFFFF)
    mm.write(off + 4, (a >> 32) & 0xFFFFFFFF)


w64(0x10, ai.physical_address)
w64(0x1c, vi.physical_address)
w64(0x28, ao.physical_address)
print(f"phys: ai=0x{ai.physical_address:x} vi=0x{vi.physical_address:x} ao=0x{ao.physical_address:x}", flush=True)
print("ctrl before start:", hex(mm.read(0x00)), flush=True)
t = time.time()
mm.write(0x00, 1)  # ap_start
done = False
while time.time() - t < 30:
    c = mm.read(0x00)
    if c & 0x2:
        done = True
        break
    time.sleep(0.05)
dt = time.time() - t
print(f"ap_done={done} after {dt:.2f}s, ctrl=0x{mm.read(0x00):x}", flush=True)
if done:
    ao.invalidate()
    out = np.array(ao)
    print(f"out range [{out.min()},{out.max()}] (zero in -> near-zero out expected)", flush=True)
print("DIAG DONE", flush=True)
