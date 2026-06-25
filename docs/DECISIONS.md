# DECISION LOG (ADR-style)

Each significant choice: what was decided, why, by whom, when. Append-only; supersede rather than
delete (mark old ones `SUPERSEDED`). Status: ✅ decided · ◇ open (owner) · 🔬 to-be-decided-by-evidence.

---

### D-1 ✅ Start a new, independent project rather than refactor the reference deployment
**2026-06-25 — owner.** The reference U-Net's 4-bitstream form factor is intrinsic to its
architecture (long time axis + U-Net skips). Escaping it requires re-architecting, not refactoring.
New repo, own git/venv/docs; old project is reference-only.

### D-2 ✅ Time-domain ONLY (for now); STFT/frequency deferred
**2026-06-25 — owner.** Phase 2 stays time-domain. The STFT-mask candidate (C3) is **out of scope**
for now (revisit only if time-domain options underdeliver on the quality-vs-fit Pareto). Rationale:
lower risk, no FPGA (i)STFT, closer to the existing know-how. Frequency domain is the biggest
departure and is parked, not killed.

### D-8 ✅ Phase-2 candidate set + open exploration mandate
**2026-06-25 — owner.** Prototype **C4 (tiled U-Net)** and **C2 (streaming TCN)**; keep **C5
(DDR-staged)** as the Plan-B control. Owner explicitly mandated exploring **additional time-domain
ideas beyond the documented candidates** ("don't be limited by the old docs; think for yourself;
more attempts encouraged"). Added by analysis (scored with the validated model): **C7 Conv-TasNet-style
time-domain mask** (removes the U-Net skip wall — the root cause — while staying time-domain),
plus **C8 recompute-skip** and **C9 compressed-skip** as combinable BRAM-reclaim levers.

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

### D-9 ✅ Phase 3 = HLS-fit-first on C7, with placeholder weights
**2026-06-25 — owner.** Take **C7 (Conv-TasNet-style)** to Phase 3. **First confirm the structure
actually fits** via real Vitis HLS C-synth + Vivado synth/P&R reports, **then** come back to iterate /
retrain a high-quality version. Key consequence: fit is structure-driven, not weight-value-driven, so
Phase 3a synthesizes C7 with **placeholder weights** to get the fit answer fast, decoupled from
quality training. (Pareto: C7 +3.79 dB SI-SDR small-run, 0.017 MB working set; C4 is the high-quality
fallback.) Toolchain on this machine is **Vitis HLS / Vivado 2022.2** (not 2024.2) — fine for the fit
check on ZU48DR.

## Pending owner gates (forward-looking)

- ~~before Phase 2: D-2~~ → resolved (D-2: time-domain only).
- ~~after Phase 1: pick directions~~ → resolved (D-8: C4 + C2 + C5, + open exploration → C7/C8/C9).
- **◇ after Phase 2**: pick the Pareto-frontier operating point (quality vs working-set).
