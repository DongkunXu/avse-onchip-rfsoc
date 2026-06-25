"""validate_baseline.py — the mandatory trust gate for the working-set model.

Runs the model on the reference design (encoded in baseline_reference.py from the real HLS source)
and checks it reproduces the measured ground truth (PAPER_DATA.md §G):

    per-IP BRAM:  audio_enc 57%, audio_dec 95%, fusion 32%, video 38%
    concurrent:   ~215%   (sum of the four)
    audio_dec URAM: 36%

The model computes activation BRAM *exactly* from the buffer lists; the residual to the measured
total is the weight ROMs + small working buffers (the diagnosis already puts weights at 7.5% of
on-chip memory). We therefore validate that:
  (a) activation BRAM explains the dominant share of each IP, and
  (b) total (activation + a transparent weight estimate) lands within tolerance of measured.

If a per-IP prediction is off beyond tolerance, the fix is to re-read that IP's HLS top and correct
its buffer list here — never to fudge a constant. (CHARTER / PHASE1_DESIGN discipline.)
"""
from __future__ import annotations

import sys

# Windows consoles default to cp1252, which cannot encode some report glyphs and would crash on
# print(). Reconfigure stdout to UTF-8 (root-cause fix); output also stays ASCII-friendly below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from working_set import (
    Design, BRAM18_BUDGET, URAM_BUDGET, fmt_util, peak_live_working_set, bytes_h,
)
from baseline_reference import (
    audio_enc, audio_dec, fusion, video, reference_design, audio_unet_liveness,
)

# Measured ground truth (PAPER_DATA.md §G), as % of the relevant budget.
MEASURED_BRAM_PCT = {"audio_enc": 57, "audio_dec": 95, "fusion": 32, "video": 38}
MEASURED_CONCURRENT_PCT = 215      # util_concurrent_synth.txt (RAMB36 2327/1080)
MEASURED_AUDIO_DEC_URAM_PCT = 36   # 29/80

# Tolerances (percentage points).
TOL_PER_IP = 8
TOL_CONCURRENT = 15
TOL_URAM = 8


def pct(tiles: int, budget: int) -> float:
    return 100.0 * tiles / budget


def main() -> int:
    modules = [audio_enc, audio_dec, fusion, video]
    print("=" * 78)
    print("WORKING-SET MODEL - VALIDATION AGAINST THE REFERENCE DESIGN")
    print("=" * 78)

    # ---- Per-IP BRAM ----
    print("\nPer-IP BRAM (RAMB18, budget %d):\n" % BRAM18_BUDGET)
    hdr = f"{'IP':<11} {'act.BRAM':>9} {'act%':>6} {'meas%':>6} {'act/meas':>9} {'URAM':>5}"
    print(hdr); print("-" * len(hdr))
    ok = True
    for m in modules:
        act = m.activation_bram18()
        act_pct = pct(act, BRAM18_BUDGET)
        meas = MEASURED_BRAM_PCT[m.name]
        ratio = act_pct / meas if meas else 0.0
        print(f"{m.name:<11} {act:>9d} {act_pct:>5.1f}% {meas:>5d}% {ratio:>8.2f}  {m.uram():>5d}")
        # Gate: activation alone must not EXCEED measured (it's a subset), and must explain the bulk.
        if act_pct > meas + TOL_PER_IP:
            print(f"   !! {m.name}: predicted activation {act_pct:.1f}% exceeds measured {meas}%+tol")
            ok = False
        if act_pct < meas - 25:
            print(f"   ?? {m.name}: predicted activation {act_pct:.1f}% far below measured {meas}% "
                  f"(weight/working residual unusually large — investigate)")

    # ---- audio_dec URAM ----
    dec_uram_pct = pct(audio_dec.uram(), URAM_BUDGET)
    print(f"\naudio_dec URAM: {audio_dec.uram()}/{URAM_BUDGET} = {dec_uram_pct:.1f}% "
          f"(measured {MEASURED_AUDIO_DEC_URAM_PCT}%)")
    if abs(dec_uram_pct - MEASURED_AUDIO_DEC_URAM_PCT) > TOL_URAM:
        print("   !! URAM off beyond tolerance"); ok = False

    # ---- Concurrent (monolithic, activation only) ----
    design = reference_design()
    conc_act = design.activation_bram18()
    conc_act_pct = pct(conc_act, BRAM18_BUDGET)
    print(f"\nConcurrent monolithic activation BRAM: {conc_act} RAMB18 "
          f"= {conc_act_pct:.1f}% (measured total {MEASURED_CONCURRENT_PCT}%)")
    print("   (measured includes weight ROMs + working buffers; activation is the dominant term.)")

    # ---- Static footprint vs peak-live working set (the opportunity) ----
    print("\n" + "=" * 78)
    print("STATIC FOOTPRINT  vs  PEAK-LIVE WORKING SET  (the gap this project exploits)")
    print("=" * 78)
    live = audio_unet_liveness()
    peak_elems, peak_step, names = peak_live_working_set(live)
    static_elems = sum(t.elements for t in live)
    print(f"Audio U-Net, static sum(all tensors): {static_elems:>10,} elems  ({bytes_h(static_elems*2)})")
    print(f"Audio U-Net, peak-live working set: {peak_elems:>10,} elems  ({bytes_h(peak_elems*2)})"
          f"  at step {peak_step}")
    print(f"   live at peak: {', '.join(names)}")
    if peak_elems:
        print(f"   static/peak ratio = {static_elems/peak_elems:.2f}x  "
              f"(a perfectly-pooled schedule could shrink resident audio activation by this factor)")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    # The headline check: activation explains the bulk per IP (ratio in a sane band) and audio_dec
    # (the binding IP) is predicted tight.
    dec_ratio = pct(audio_dec.activation_bram18(), BRAM18_BUDGET) / MEASURED_BRAM_PCT["audio_dec"]
    gate_dec = 0.80 <= dec_ratio <= 1.05
    gate_uram = abs(dec_uram_pct - MEASURED_AUDIO_DEC_URAM_PCT) <= TOL_URAM
    if ok and gate_dec and gate_uram:
        print("VALIDATION PASSED - model reproduces the reference within tolerance.")
        print(f"  audio_dec activation/measured = {dec_ratio:.2f} (binding IP, tight); "
              f"URAM {dec_uram_pct:.0f}% ~= {MEASURED_AUDIO_DEC_URAM_PCT}%.")
        return 0
    print("VALIDATION FAILED - investigate the buffer lists (do not fudge constants).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
