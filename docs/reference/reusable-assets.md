# Migrated reusable assets — inventory & provenance

What was carried over from the reference project (`../UNet-AVSE-Vitis`), why, and what was adapted.
Goal: avoid re-doing solid, architecture-agnostic work, **without** importing the old project's
architectural assumptions.

| New location | Source | Treatment | Notes |
|---|---|---|---|
| `src/avse/data/dataset.py` | `training/data/dataset.py` | verbatim | High-quality, Windows-friendly, parallel, already window/streaming-based. Reads LRS3 via `config.data.root_dir`. |
| `src/avse/data/data_module.py` | `training/data/data_module.py` | verbatim | Lightning `DataModule` (train/dev). |
| `src/avse/config/config.py` | `training/config/config.py` | verbatim | Dataclass + yaml loader (`Config`, `DataConfig`, ...). |
| `src/avse/config/reference_base_config.yaml` | `training/config/base_config.yaml` | verbatim | The OLD U-Net's Tier-D/E config — kept as reference. |
| `src/avse/config/onchip_config.yaml` | (new) | written | Platform-stable config: dataset path = `D:/DataSet/LRS3`, audio/video/window. `model:` left TBD. |
| `src/avse/metrics/audio_metrics.py` | `dfx/board_test/score_fpga_vs_pytorch.py` | extracted + cleaned | Pure metric functions (SI-SDR / SNR / cos / PESQ-WB / STOI), file-I/O removed. Architecture-agnostic. |
| `src/avse/losses/losses.py` | `training/utils/losses.py` | verbatim | `ImprovedAVSELoss` (time + STFT + perceptual). Depends on `asteroid` + `torch_pesq`. May be replaced. |
| `src/avse/reference/*.py` | `training/models/*.py` | verbatim | The 0.37 M teacher (audio/video encoders, fusion, top model). **Distillation/comparison ONLY.** |

## Deliberately NOT migrated (yet)

- **New model architectures** — written fresh in `src/avse/models/` once Phase 1 picks directions.
  The old U-Net is *not* copied there by design.
- **HLS C++ / Vivado / board scripts** — Phase 3 only; documented in `hls/`, `hw/`, and
  [`hardware-budget.md`](hardware-budget.md) / [`prior-wins.md`](prior-wins.md), not copied.
- **Dataset prep scripts** — they live with the dataset at `D:\DataSet\LRS3` (MixWithSNR.py,
  Separate_video_audio.py, preprocess_video_96.py, ...). No need to duplicate.

## Known caveats / to-fix-when-needed

- `src/avse/reference/avse_model.py` imports `from utils.losses import ImprovedAVSELoss` (old layout).
  In this project the loss is `avse.losses.ImprovedAVSELoss`. The sub-encoders import cleanly; only the
  top LightningModule needs this one line fixed if the full teacher is instantiated (for distillation).
- `losses.py` uses `torch.cuda.amp.autocast(enabled=False)` (deprecated in newer torch). Harmless now;
  modernize to `torch.amp.autocast('cuda', ...)` if a deprecation becomes an error.
- `asteroid` pulled an older `torchmetrics` (0.11.4) into the venv; installed cleanly but watch for
  friction with torch 2.11 at Phase-2 runtime.
