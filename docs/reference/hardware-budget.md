# Reference fact: device budget (ZU48DR / xczu48dr-ffvg1517-2-e)

The single-board target. All Phase-1 fit estimates and Phase-3 reports are measured against this.

| Resource | Budget | Notes |
|---|---|---|
| BRAM | **1080 tiles** (= 2160 × BRAM_18K, ≈ 38 Mb) | **the binding resource** for this workload |
| URAM | **80** (≈ 23 Mb) | 78 % unused on the reference design — untapped if 16-bit data is packed 4-per-72-bit word |
| DSP | **4272** | only ~31 % used concurrently — plenty of compute headroom |
| LUT | **425,280** | second constraint (reference concurrent = 126 %) |
| FF | **850,560** | ~52 % used — not a constraint |
| DDR4 | on-board, **GBs** | essentially unused by the compute path — available for off-chip activation staging |
| AI Engines | **none** | Versal-only; not available here |
| HBM | **none** | — |

## How the budget frames the problem

- **On-chip memory ceiling** ≈ 38 Mb BRAM + 23 Mb URAM ≈ **61 Mb ≈ 7.6 MB** of fast on-chip storage,
  *if* both are used efficiently. The reference design's resident activation set alone (~2.2 MB of
  skips, before working buffers and weight ROMs) plus 4-way duplication is what busts this.
- **Compute is NOT the constraint** (DSP 31 %, FF 52 %). This is a *memory-and-dataflow* problem. There
  is room to spend more compute (e.g. recompute-instead-of-store) to buy memory.
- **Latency is NOT the constraint** (RTF 0.468 — half the real-time budget idle). Serialization and
  off-chip round-trips that cost latency but save BRAM are cheap.

## Real-time budget (for reference)

| Quantity | Value |
|---|---|
| Window | 1.2 s @ 16 kHz = 19200 samples + 30 frames 96×96 |
| Reference compute time | 561.25 ms / window (deterministic) |
| Compute RTF | **0.468** (≈ 2× real-time headroom) |
