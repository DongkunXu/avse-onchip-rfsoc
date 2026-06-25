# Phase 1 — design of the analytical working-set / BRAM model

This note records *how* the model works and *why* it is trustworthy, so the Phase-1 scoring is
defensible (it is the quantitative spine of the circuits paper). Written before the code so the
design is deliberate, not reverse-justified.

## What we are modelling, and the key realisation

The binding resource is **on-chip BRAM**. The crucial fact about HLS:

> HLS arrays are **statically allocated**. A design's BRAM ≈ **Σ over ALL arrays it declares**
> (× partition banking, × ping-pong if DATAFLOW), *not* the dynamic peak-live set — unless arrays
> are explicitly reused/pooled.

So there are **two** numbers worth computing, and their gap is the whole opportunity:

1. **Static-allocation footprint** — Σ of every declared buffer. This is what naive HLS (and the
   reference design) pays. It is what we validate against the measured 215 % / 95 %.
2. **Peak-live working set** — Σ C×T over only the *simultaneously-live* tensors, via liveness over
   the execution schedule. This is the *floor* a smart pooled/streaming design could reach. It is the
   architecture-level metric for ranking candidates.

A single-static-config design fits iff its **static footprint** ≤ budget. The research goal is to
choose an architecture + dataflow whose footprint (ideally approaching its peak-live set) fits.

## Ground truth used to calibrate (from the reference HLS source + reports)

- Data type: `data_t = ap_fixed<16,7>` → **16-bit** activations (`src/common/types.hpp`).
- `audio_decoder_top.cpp` (the 95 %-BRAM IP) declares, with DATAFLOW **dropped** (no ping-pong):
  - input staging: `bottleneck[192][600]`→**URAM**; `skip0[32][9600]`, `skip1[64][4800]`,
    `skip2[96][2400]` all `cyclic factor=2`; `skip3[128][1200]` factor 1.
  - per-stage: `s0_main[128][1200]`, `s1_main[96][2400]`, `s2_main[64][4800]`,
    `s3_main[32][9600]` all `cyclic factor=2`; `s4_out[1][19200]` factor 1.
  - 27 weight ROMs, partitioned `dim=2 complete` (mostly LUTRAM, small BRAM residue).
- Measured (PAPER_DATA §G, of 2160 BRAM_18K): audio_enc 57 %, **audio_dec 95 %**, fusion 32 %,
  video 38 %. Concurrent monolithic = **215 %** (sum ≈ 222 %). audio_dec URAM = 36 % (29/80).

## The BRAM mapping (first-principles, then calibrated)

For a buffer `[C][T]` of `b`-bit data with cyclic partition factor `P` on the channel dim:

```
banks            = P
words_per_bank   = ceil(C / P) * T
RAMB18_per_bank  = ceil(words_per_bank / WORDS_PER_RAMB18)      # depth packing
BRAM18(buffer)   = banks * RAMB18_per_bank * ceil(b / 18)       # width packing
```

- `WORDS_PER_RAMB18 = 1024` (a RAMB18E2 = 18 Kb → 1K×18 for ≤18-bit data).
- URAM: `URAM = 4096×72b`; naive 16-bit storage = `ceil(C*T / 4096)` URAMs (78 % waste; matches the
  29-URAM bottleneck offload). A packed mode (4×16b/word) divides this by ~4.

Sanity check (hand-computed, activations only, P from the source):
audio_dec ≈ 1971 BRAM18 ≈ 91 % → +weight residue ≈ **95 %** ✅;
audio_enc skips+input ≈ 1109 BRAM18 ≈ 51 % → +weights ≈ **57 %** ✅.

The two free knobs (`WORDS_PER_RAMB18`, a small per-IP weight-ROM BRAM residue) are fixed once,
against these measured numbers, in `validate_baseline.py`. If a per-IP prediction is off by more than
a few points, we **read that IP's HLS top and fix the buffer list** — not fudge the knob.

## Deliverables

| File | Role |
|---|---|
| `working_set.py` | Core: `Tensor`/`Buffer`/`Module`/`Design`, BRAM/URAM mapping, liveness peak. |
| `baseline_reference.py` | The reference 4-IP design encoded as data (buffer plans from the HLS source). |
| `validate_baseline.py` | Run model on baseline; assert audio_dec ≈ 95 %, concurrent ≈ 215 %; report static-vs-peak gap. **The mandatory trust gate.** |
| `candidates/` | Later: each candidate architecture as a `Design`, scored by the *same* model. |

## Validation gate (must pass before any candidate is scored)

1. Per-IP BRAM within ~±5 pts of {57, 95, 32, 38}.
2. Concurrent ≈ 215 % (±10 pts).
3. audio_dec URAM ≈ 36 %.
Only then is the model trusted to rank unbuilt candidates.
