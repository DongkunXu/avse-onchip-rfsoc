# DECISION LOG (ADR-style)

Each significant choice: what was decided, why, by whom, when. Append-only; supersede rather than
delete (mark old ones `SUPERSEDED`). Status: ✅ decided · ◇ open (owner) · 🔬 to-be-decided-by-evidence.

---

### D-1 ✅ Start a new, independent project rather than refactor the reference deployment
**2026-06-25 — owner.** The reference U-Net's 4-bitstream form factor is intrinsic to its
architecture (long time axis + U-Net skips). Escaping it requires re-architecting, not refactoring.
New repo, own git/venv/docs; old project is reference-only.

### D-2 ◇ Time-domain vs STFT / frequency-domain — OPEN (owner call before Phase 2)
Moving to an STFT-domain mask is the most fundamental (root-cause) change — it makes
frame-synchronous streaming natural — but the biggest departure from the reference. Staying
time-domain weights the dataflow axes (streaming / tiling / DDR) instead. **Phase 1 will score both
families; the owner chooses before Phase 2 training begins.**

### D-3 ✅ Precision locked at int16
**Inherited.** int8/DPU costs −1.6 to −2.9 dB SI-SDR and breaks the quality floor; QAT judged
unlikely to recover enough. See [`reference/dead-ends.md`](reference/dead-ends.md). 16-bit activations
are *why* BRAM dominates — that constraint is accepted, not fought with lower precision.

### D-4 ✅ Single board fixed: RFSoC 4x2 / ZU48DR
**Inherited.** No AI Engines (Versal-only). AIE/Versal offload is a different project, revisited only
if the owner changes hardware.

### D-5 ✅ Binding metric = peak simultaneously-live activation working set (Σ C×T)
**2026-06-25.** This single, analytically-computable quantity is the optimization target and the
spine of the circuits contribution. Phase 1 builds and validates a model for it.

### D-6 ✅ torch from the cu128 wheel index (not the inherited cu121 pins)
**2026-06-25.** Host GPU is RTX 5070 Ti (Blackwell, sm_120); cu121 builds do not support it. The new
venv installs torch/torchvision/torchaudio from `https://download.pytorch.org/whl/cu128`.

### D-7 ✅ Reusable assets migrated, not re-derived; new models written fresh
**2026-06-25.** Architecture-agnostic, high-value code (data pipeline, audio metrics, loss, config
template, teacher-model reference) was migrated to avoid re-doing solid work. **New, hardware-shaped
model architectures are written from scratch** in `src/avse/models/` — the migrated teacher lives in
`src/avse/reference/` for distillation/comparison only, never as the design center.

---

## Pending owner gates (forward-looking)

- **◇ before Phase 2 start**: D-2 (time-domain vs STFT).
- **◇ after Phase 1**: pick the 2–3 candidate directions from the scoring table.
- **◇ after Phase 2**: pick the Pareto-frontier operating point.
