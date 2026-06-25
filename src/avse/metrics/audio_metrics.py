"""Speech-enhancement quality metrics — clean numpy implementations.

Extracted and consolidated from the reference project's scoring harness
(`dfx/board_test/score_fpga_vs_pytorch.py`), with the file-I/O stripped out so these are pure,
reusable functions. Architecture-agnostic: they take float waveforms and return scalars, so they
serve every Phase-2 candidate identically.

Conventions:
    - waveforms are 1-D float numpy arrays (or array-likes), nominally in [-1, 1]
    - `est` = estimate / enhanced output, `ref` = clean target reference
    - PESQ/STOI return None when a signal is silent or the optional dep is unavailable
"""
from __future__ import annotations

from typing import Optional, Dict
import numpy as np

SAMPLE_RATE = 16000
_EPS = 1e-9


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def cos_sim(a, b, eps: float = _EPS) -> float:
    """Cosine similarity between two signals (1.0 == identical direction)."""
    a, b = _as_1d(a), _as_1d(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def si_sdr(est, ref, eps: float = _EPS) -> float:
    """Scale-invariant SDR in dB. Mean-centres, projects est onto ref, then 10log10(s/e)."""
    est, ref = _as_1d(est), _as_1d(ref)
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s = alpha * ref
    e = est - s
    return float(10.0 * np.log10((np.dot(s, s) + eps) / (np.dot(e, e) + eps)))


def snr(est, ref, eps: float = _EPS) -> float:
    """Plain SNR in dB treating `ref` as the clean signal and (est-ref) as noise."""
    est, ref = _as_1d(est), _as_1d(ref)
    noise = est - ref
    return float(10.0 * np.log10((np.dot(ref, ref) + eps) / (np.dot(noise, noise) + eps)))


def pesq_wb(ref, est, sr: int = SAMPLE_RATE) -> Optional[float]:
    """Wide-band PESQ (target=ref, degraded=est). None if silent or `pesq` unavailable."""
    ref, est = _as_1d(ref), _as_1d(est)
    if np.max(np.abs(est)) < 1e-6 or np.max(np.abs(ref)) < 1e-6:
        return None
    try:
        from pesq import pesq
        v = float(pesq(sr, ref, est, "wb"))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def stoi_score(ref, est, sr: int = SAMPLE_RATE, extended: bool = False) -> Optional[float]:
    """STOI (or ESTOI if extended=True). None if silent or `pystoi` unavailable."""
    ref, est = _as_1d(ref), _as_1d(est)
    if np.max(np.abs(est)) < 1e-6 or np.max(np.abs(ref)) < 1e-6:
        return None
    try:
        from pystoi import stoi
        v = float(stoi(ref, est, sr, extended=extended))
        if not np.isfinite(v):
            return None
        # pystoi returns exactly 1e-5 as a sentinel when a window has too few STFT frames after
        # silent-frame removal (it warns). Treat that as "not measurable" rather than a real 0 score.
        if abs(v - 1e-5) < 1e-12:
            return None
        return v
    except Exception:
        return None


def evaluate_pair(est, ref, sr: int = SAMPLE_RATE) -> Dict[str, Optional[float]]:
    """Convenience: all standard metrics for one (estimate, reference) pair."""
    return {
        "si_sdr": si_sdr(est, ref),
        "snr": snr(est, ref),
        "cos_sim": cos_sim(est, ref),
        "pesq_wb": pesq_wb(ref, est, sr),
        "stoi": stoi_score(ref, est, sr),
    }
