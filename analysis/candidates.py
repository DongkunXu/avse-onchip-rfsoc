"""candidates.py — candidate architectures, scored by the *validated* working-set model.

Each candidate is expressed as a liveness schedule (or a closed-form peak) so its **peak-live
activation working set** — the metric validated in validate_baseline.py — is computed the same way
for every approach. Peak-live (MB) is the honest, apples-to-apples discriminator; rough BRAM% and the
qualitative risk/effort columns are advisory.

Design center (CHARTER §3): minimise peak simultaneously-live activation so the whole AVSE co-resides
on-chip in ONE static configuration. The reference U-Net peak-live is 2.4 MB (audio) — already near
the whole BRAM budget — so a winner must *bound the live temporal extent*, not just schedule.

All candidates target the same task: 1.2 s @ 16 kHz (19200 samples) + 30 frames 96x96.
Channel/stride schedule reused from the reference where a candidate keeps a U-Net spine:
  encoder C = [32,64,96,128,192], T = 19200/2^l ; video ~ 30x96 (cheap, shared by all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math

from working_set import LiveTensor, peak_live_working_set, BRAM18_BUDGET

# Reference audio U-Net per-stage (C, T_out) — the spine several candidates reuse.
ENC = [(32, 9600), (64, 4800), (96, 2400), (128, 1200), (192, 600)]  # skip0..3 + bottleneck
FULL_T = 19200
VIDEO_FEAT = 30 * 96          # cheap shared video embedding (elements)


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────
def unet_peak_elems(time_scale: float = 1.0) -> int:
    """Peak-live elements of the reference U-Net schedule with every T multiplied by `time_scale`.

    time_scale = 1.0 -> full window (matches the validated 1.27M). 0.125 -> 1/8 temporal tile, etc.
    Skips co-reside across encode->decode; this is the structural floor of the U-Net topology.
    """
    def ts(t):  # scale + floor to >=1
        return max(1, int(round(t * time_scale)))
    live = [
        LiveTensor("in_audio", 1, ts(FULL_T), 0, 0),
        LiveTensor("skip0", 32,  ts(9600), 0, 9),
        LiveTensor("skip1", 64,  ts(4800), 1, 8),
        LiveTensor("skip2", 96,  ts(2400), 2, 7),
        LiveTensor("skip3", 128, ts(1200), 3, 6),
        LiveTensor("bottleneck", 192, ts(600), 4, 6),
        LiveTensor("dec_s0", 128, ts(1200), 6, 7),
        LiveTensor("dec_s1", 96,  ts(2400), 7, 8),
        LiveTensor("dec_s2", 64,  ts(4800), 8, 9),
        LiveTensor("dec_s3", 32,  ts(9600), 9, 10),
        LiveTensor("dec_s4", 1,   ts(FULL_T), 10, 10),
    ]
    peak, _, _ = peak_live_working_set(live)
    return peak


def elems_to_bram_pct(elems: int, avg_partition: int = 2) -> float:
    """Rough BRAM% if `elems` int16 are held on-chip with average cyclic-partition banking.

    Mirrors working_set.Buffer.bram18 at an aggregate level: banking rounds each bank up to a
    RAMB18 (1024 words). Using an effective overhead for partition=2 -> ~1.3x vs ideal packing.
    """
    overhead = 1.0 + 0.15 * (avg_partition)        # ~1.3x at partition=2
    ramb18 = math.ceil(elems / 1024) * overhead
    return 100.0 * ramb18 / BRAM18_BUDGET


# Shared, roughly-fixed overhead that EVERY single-config design also carries on top of the audio
# path: the video encoder (~38% in the reference) + cross-modal fusion (~32%) + weight ROMs / working
# buffers, lightly deduped. The audio path is what candidates change; this is the rest of the system
# it must leave room for. Conservative (assumes video/fusion stay ~as-is; both are secondary levers).
SHARED_OVERHEAD_PCT = 65.0


@dataclass
class Candidate:
    key: str
    name: str
    family: str                 # "time" | "freq" | "hybrid"
    axis: str                   # which CHARTER axes it exercises
    peak_live_elems: int        # audio peak-live (+ shared video where relevant)
    reuses_weights: bool        # can it inherit the trained 0.37M teacher directly?
    quality_risk: str           # low | moderate | high  (+ one-line reason)
    effort: str                 # low | moderate | high
    notes: str = ""
    in_scope: bool = True       # False -> excluded by an owner decision (e.g. D-2), shown for record
    combinable_lever: bool = False  # True -> a BRAM-reclaim lever best COMBINED, not a standalone arch

    @property
    def peak_live_mb(self) -> float:
        return self.peak_live_elems * 2 / (1024 * 1024)

    @property
    def bram_pct(self) -> float:
        """Audio-path peak-live BRAM% (the part the candidate changes)."""
        return elems_to_bram_pct(self.peak_live_elems)

    @property
    def system_bram_pct(self) -> float:
        """Whole single-config design: audio path + the shared video/fusion/weight overhead."""
        return self.bram_pct + SHARED_OVERHEAD_PCT

    @property
    def fit_verdict(self) -> str:
        # Verdict is on the WHOLE system (must co-reside on-chip in one static config).
        p = self.system_bram_pct
        if p < 80:
            return "fits"
        if p < 100:
            return "borderline"
        return "DOES NOT FIT"


# ── the candidate set ─────────────────────────────────────────────────────────────────────────────
def all_candidates() -> List[Candidate]:
    cands: List[Candidate] = []

    # C0 — Reference full-window U-Net (the baseline; single-config does NOT fit).
    cands.append(Candidate(
        "C0", "Reference full-window U-Net (single-config)", "time", "(none) baseline",
        peak_live_elems=unet_peak_elems(1.0),
        reuses_weights=True, quality_risk="n/a (the reference quality)", effort="n/a",
        notes="Static footprint 215% BRAM as 4 bitstreams. Peak-live 2.4MB still ~near budget."))

    # C6 — Single-engine + global activation pool (Axis 3 ONLY): same topology, time-shared engine,
    #      one pool sized to peak-live. Quantifies that scheduling alone is insufficient.
    cands.append(Candidate(
        "C6", "Single-engine + global pool (same U-Net)", "time", "Axis 3 (schedule only)",
        peak_live_elems=unet_peak_elems(1.0),
        reuses_weights=True, quality_risk="low (math-identical to reference)", effort="high",
        notes="Pools to peak-live 2.4MB. Hand-built offset-addressed engine (path3 validated the "
              "datapath). Bounds BRAM to peak-live but that is itself ~near budget -> not enough alone."))

    # C5 — DDR-staged U-Net (Axis 2): skips to DDR, read back in decoder. On-chip = one stage + halo.
    #      Math-identical; the safe control / Plan B. Keeps model complexity + needs DDR bandwidth.
    ddr_onchip = max(c * t for c, t in ENC) + 192 * 600  # largest single stage + bottleneck working
    cands.append(Candidate(
        "C5", "DDR-staged U-Net (skips to DDR)", "time", "Axis 2 (off-chip staging)",
        peak_live_elems=ddr_onchip,
        reuses_weights=True, quality_risk="low (bit-identical to reference)", effort="moderate",
        notes="On-chip holds only the active stage; skips live in DDR (double-buffered, latency "
              "spare). Lowest quality risk; but keeps full compute + depends on DDR BW. Plan-B/control."))

    # C4 — Tiled full U-Net (Axis 2): overlapping temporal tiles with halo. Peak = one tile's U-Net.
    #      Reuses the trained model; near-lossless with correct halo.
    TILE = 1 / 8
    cands.append(Candidate(
        "C4", "Tiled U-Net (temporal tiles + halo)", "time", "Axis 2 (temporal tiling)",
        peak_live_elems=unet_peak_elems(TILE),
        reuses_weights=True, quality_risk="low (math-identical w/ correct halo; boundary care)",
        effort="moderate",
        notes=f"1/{int(1/TILE)} window tiles. Peak scales with tile length. Reuses reference weights; "
              "halo handles conv receptive fields. Strong low-risk fit route."))

    # C1 — Streaming/chunked U-Net (Axis 1+2): frame-synchronous chunks, bounded carried state.
    #      Like tiling but causal/streaming (carry overlap state), peak decoupled from window length.
    cands.append(Candidate(
        "C1", "Streaming chunked U-Net (frame-sync)", "time", "Axis 1+2 (streaming)",
        peak_live_elems=unet_peak_elems(1 / 16),
        reuses_weights=False, quality_risk="moderate (causal chunking; needs retrain/distill)",
        effort="moderate",
        notes="Process ~75ms chunks with carried receptive-field state. Peak decoupled from the 1.2s "
              "window. Pairs with distillation from the 0.37M teacher."))

    # C2 — Streaming TCN (Axis 1): dilated causal conv stack, fixed width, per-layer ring-buffer state.
    #      Peak = Σ_layers C * kernel * dilation (the conv history), independent of window.
    C_tcn, K, dil = 128, 3, [1, 2, 4, 8, 16, 32, 64, 128]
    tcn_state = sum(C_tcn * K * d for d in dil) + C_tcn * 800  # history + one 50ms frame of activations
    cands.append(Candidate(
        "C2", "Streaming TCN (dilated causal)", "time", "Axis 1 (bounded receptive field)",
        peak_live_elems=tcn_state,
        reuses_weights=False, quality_risk="moderate (strong SE family; new arch, distillable)",
        effort="moderate",
        notes=f"C={C_tcn}, {len(dil)} dilated layers. State = Σ C*K*dilation ({tcn_state:,} elems). "
              "O(state) not O(sequence). Owner prior art cites LSTM-AVSE; TCN is the conv analogue."))

    # C3 — STFT-domain mask (Axis 1): predict a T-F mask on the spectrogram, frame-synchronous.
    #      Per-frame peak = C * F (one spectral column); biggest departure (D-2), iSTFT on FPGA.
    F_bins, C_stft = 257, 32           # n_fft=512 -> 257 bins
    stft_per_frame = C_stft * F_bins * 4   # a few conv layers' worth over one frame's freq column
    cands.append(Candidate(
        "C3", "STFT-domain mask (frame-sync)", "freq", "Axis 1 (representation change)",
        peak_live_elems=stft_per_frame,
        reuses_weights=False, quality_risk="high (biggest departure; iSTFT/overlap-add on FPGA)",
        effort="high", in_scope=False,
        notes="OUT OF SCOPE per D-2 (time-domain only for now). Kept for record: operate on spectrogram "
              f"[{F_bins} x ~75], predict T-F mask frame-by-frame -> peak ~C*F ({stft_per_frame:,} elems). "
              "Root-cause fix but needs FPGA STFT/iSTFT. Revisit only if time-domain underdelivers."))

    # ───────────────────────────────────────────────────────────────────────────────────────────
    #  New, self-derived time-domain candidates (owner mandate D-8: think beyond the old docs).
    #  All attack the validated root cause — the U-Net skip-residency wall — without leaving time.
    # ───────────────────────────────────────────────────────────────────────────────────────────

    # C7 — Conv-TasNet-style time-domain mask: encoder (1 strided conv -> single-resolution latent),
    #      TCN separator predicting a mask, decoder (transposed conv). NO multi-resolution U-Net skips
    #      at all -> the skip-residency wall simply does not exist. The most direct time-domain attack
    #      on the root cause. Proven, strong SE quality; streams naturally (single resolution).
    N_tas, stride_tas = 128, 16
    T_lat = FULL_T // stride_tas                       # 1200 latent frames
    tas_streamed = N_tas * 256                         # TCN ring state over ~256 latent frames (~256ms)
    tas_full = 3 * N_tas * T_lat                       # non-streamed: latent + mask + masked
    cands.append(Candidate(
        "C7", "Conv-TasNet-style time mask (no U-Net skips)", "time", "Axis 1 (remove skip topology)",
        peak_live_elems=tas_streamed,
        reuses_weights=False, quality_risk="moderate (proven SE family; new arch, distillable)",
        effort="moderate-high",
        notes=f"encoder(stride {stride_tas}) -> latent[{N_tas} x {T_lat}] -> TCN mask -> decoder. "
              f"NO multi-resolution skips -> NO co-residency wall. Streamed peak ~{tas_streamed:,} elems; "
              f"even non-streamed ~{tas_full:,} (~{tas_full*2/1e6:.1f}MB) with no skip wall. Time-domain, "
              "frame-synchronous, distillable from the 0.37M teacher. STRONG root-cause candidate."))

    # C8 — Recompute-skip U-Net (lever): keep the U-Net, but DON'T store the two shallow skips (skip0
    #      32x9600, skip1 64x4800 = 55% of resident); regenerate them in the decoder by re-running
    #      enc layers 0-1 from a stored coarse checkpoint. Trades compute (DSP 31%, 2x latency spare)
    #      for BRAM (the wall) — exploits the exact resource asymmetry. Best COMBINED with tiling.
    recompute_resident = (96 * 2400) + (128 * 1200) + (192 * 600)  # only deep skips + bottleneck kept
    cands.append(Candidate(
        "C8", "Recompute-skip U-Net (regenerate shallow skips)", "time", "Axis 2 (recompute-vs-store)",
        peak_live_elems=recompute_resident,
        reuses_weights=True, quality_risk="low (bit-identical; pure schedule/recompute)",
        effort="moderate-high", combinable_lever=True,
        notes="Don't store skip0/skip1 (55% of resident); recompute them on demand from a coarse "
              "checkpoint. Trades cheap compute for scarce BRAM. ~half the skip residency. Borderline "
              "alone; shines COMBINED with tiling/streaming. Reuses the trained weights."))

    # C9 — Compressed-skip U-Net (lever): store the shallow skips at reduced channel count via a 1x1
    #      bottleneck (skip0 32->8, skip1 64->16), expand in the decoder. Directly shrinks the
    #      dominant C*T term. Small quality cost; needs a light retrain of the bottlenecks.
    compressed_resident = (8 * 9600) + (16 * 4800) + (96 * 2400) + (128 * 1200) + (192 * 600)
    cands.append(Candidate(
        "C9", "Compressed-skip U-Net (learned skip bottleneck)", "time", "Axis 1+4 (capacity knob)",
        peak_live_elems=compressed_resident,
        reuses_weights=False, quality_risk="low-moderate (slight loss from skip compression)",
        effort="moderate", combinable_lever=True,
        notes="Store shallow skips at reduced channels (32->8, 64->16) -> attacks the dominant C*T "
              "directly (~40% less resident). Borderline alone; combine with tiling. Light retrain. "
              "Also viable but not separately scored: sub-band multirate (learned filterbank) and "
              "dual-path chunked-recurrent (DPRNN-style) — both time-domain, bounded peak."))

    return cands
