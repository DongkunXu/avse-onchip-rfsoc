"""working_set.py — the analytical on-chip-memory model (Phase 1 spine).

Computes, for any AVSE architecture expressed as buffers + a schedule, the two numbers that decide
whether it can run in a single static FPGA configuration (see ../docs/CHARTER.md §3 and
PHASE1_DESIGN.md):

  (1) STATIC-ALLOCATION FOOTPRINT  — Σ over ALL declared buffers (× partition banking, × ping-pong).
      This is what naive HLS pays, and what a single-static-config bitstream must fit. Validated
      against the reference design's measured 215 % / 95 % BRAM.

  (2) PEAK-LIVE WORKING SET         — Σ C×T over only simultaneously-live tensors, via liveness over
      the execution schedule. The floor a pooled/streaming design could reach. The metric for
      ranking candidate architectures.

The gap between (1) and (2) is the opportunity this project exploits.

Units are kept explicit. The BRAM model maps a buffer [C][T] of b-bit data with cyclic partition
factor P (on the channel dim) to RAMB18 tiles; see `Buffer.bram18`. Calibration of the one capacity
constant is validated in validate_baseline.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── Device budget: ZU48DR / xczu48dr-ffvg1517-2-e (docs/reference/hardware-budget.md) ──────────
BRAM18_BUDGET = 2160          # RAMB18 tiles (= 1080 RAMB36)
URAM_BUDGET = 80
DSP_BUDGET = 4272
LUT_BUDGET = 425_280

# ── BRAM/URAM physical capacity (the calibrated constants) ─────────────────────────────────────
# A RAMB18E2 = 18 Kb. For <=18-bit data the natural config is 1K x 18 -> 1024 words/tile.
WORDS_PER_RAMB18 = 1024
RAMB18_WIDTH = 18
# A URAM = 4K x 72b. Storing one 16-bit value per 72-bit word wastes ~78 % (the reference's
# bottleneck offload needed 29 URAMs for 115200 int16 -> 115200/4096 = 28.1). A packed layout
# (4 x 16b per word) divides the count by ~4.
URAM_WORDS_NAIVE_16B = 4096
URAM_PACK_16B = 4             # 16-bit values packable per 72-bit word (4*16=64<=72)


@dataclass(frozen=True)
class Buffer:
    """One statically-allocated on-chip array, modelled after an HLS `static data_t name[C][T]`.

    partition : cyclic ARRAY_PARTITION factor on the channel (dim=1) axis -> number of banks.
    storage   : "bram" | "uram".
    pingpong  : True if a top-level DATAFLOW PIPO doubles this buffer.
    uram_packed : if storage=="uram", whether 4x16b packing is assumed (else naive 1-per-word).
    """
    name: str
    C: int
    T: int
    bits: int = 16
    partition: int = 1
    storage: str = "bram"
    pingpong: bool = False
    uram_packed: bool = False

    @property
    def elements(self) -> int:
        return self.C * self.T

    @property
    def bytes(self) -> int:
        return self.elements * self.bits // 8

    def bram18(self) -> int:
        if self.storage != "bram":
            return 0
        banks = max(1, self.partition)
        words_per_bank = math.ceil(self.C / banks) * self.T
        depth_tiles = math.ceil(words_per_bank / WORDS_PER_RAMB18)
        width_tiles = math.ceil(self.bits / RAMB18_WIDTH)
        n = banks * depth_tiles * width_tiles
        return n * (2 if self.pingpong else 1)

    def uram(self) -> int:
        if self.storage != "uram":
            return 0
        per_word = URAM_PACK_16B if self.uram_packed else 1
        words = math.ceil(self.elements / per_word)
        n = math.ceil(words / (URAM_WORDS_NAIVE_16B))
        # (URAM is 4096 deep regardless of packing; packing reduces logical word count first.)
        return n * (2 if self.pingpong else 1)


@dataclass
class Module:
    """One IP / processing stage: its activation buffers + a weight-ROM BRAM estimate.

    weight_bram18 : RAMB18 the weight ROMs land in. Most weights are `dim=2 complete`-partitioned
                    and map to LUTRAM, so this is small; it is the documented residual between the
                    (exact) activation footprint and the measured total. Pass 0 to report
                    activation-only.
    """
    name: str
    buffers: List[Buffer] = field(default_factory=list)
    weight_bram18: int = 0
    weight_params: int = 0          # for the param-footprint accounting (informational)
    note: str = ""

    def activation_bram18(self) -> int:
        return sum(b.bram18() for b in self.buffers)

    def bram18(self) -> int:
        return self.activation_bram18() + self.weight_bram18

    def uram(self) -> int:
        return sum(b.uram() for b in self.buffers)

    def activation_elements(self) -> int:
        return sum(b.elements for b in self.buffers)


@dataclass
class Design:
    """A full system = a set of modules that must co-reside in ONE static configuration."""
    name: str
    modules: List[Module] = field(default_factory=list)

    # ---- static-allocation footprint (the single-config fit test) ----
    def bram18(self) -> int:
        return sum(m.bram18() for m in self.modules)

    def activation_bram18(self) -> int:
        return sum(m.activation_bram18() for m in self.modules)

    def uram(self) -> int:
        return sum(m.uram() for m in self.modules)

    def bram_util(self) -> float:
        return self.bram18() / BRAM18_BUDGET

    def uram_util(self) -> float:
        return self.uram() / URAM_BUDGET

    def fits(self) -> bool:
        return self.bram18() <= BRAM18_BUDGET and self.uram() <= URAM_BUDGET


# ────────────────────────────────────────────────────────────────────────────────────────────────
#  Peak-live working set (liveness) — the candidate-ranking metric
# ────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class LiveTensor:
    """A logical tensor with a lifetime over an integer schedule [produced, last_used]."""
    name: str
    C: int
    T: int
    produced: int      # schedule step at which it becomes live
    last_used: int     # last schedule step at which it is read (inclusive)
    bits: int = 16

    @property
    def elements(self) -> int:
        return self.C * self.T


def peak_live_working_set(tensors: List[LiveTensor]) -> Tuple[int, int, List[str]]:
    """Sweep the schedule; return (peak_elements, peak_step, names_live_at_peak).

    A tensor is live on steps [produced, last_used] inclusive. The peak is the step maximising the
    sum of C×T over all live tensors — the minimum activation memory a perfectly-pooled
    implementation of this schedule would still need to hold at once.
    """
    if not tensors:
        return 0, 0, []
    steps = range(min(t.produced for t in tensors), max(t.last_used for t in tensors) + 1)
    best = (-1, 0, [])
    for s in steps:
        live = [t for t in tensors if t.produced <= s <= t.last_used]
        total = sum(t.elements for t in live)
        if total > best[0]:
            best = (total, s, [t.name for t in live])
    return best


def bytes_h(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} B"
        n_bytes /= 1024


def fmt_util(tiles: int, budget: int) -> str:
    return f"{tiles:>5d} / {budget} ({100*tiles/budget:5.1f}%)"
