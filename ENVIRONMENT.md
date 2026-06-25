# ENVIRONMENT

Everything needed to reproduce the working environment for this project on this machine.

## Host

| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| Project root | `G:\phD_Projects\AVSE-OnChip-RFSoC` |
| GPU | **NVIDIA RTX 5070 Ti — Blackwell, sm_120, 16 GB**, driver 610.47 |
| Python (training) | 3.11.9 at `C:\Users\dongk\AppData\Local\Programs\Python\Python311\python.exe` |

## Python virtual environment

The project owns its own venv at `./.venv` (git-ignored). Create it with:

```bash
bash tools/setup_venv.sh        # logs to tools/setup_venv.log
```

### ⚠️ The torch / Blackwell gotcha (do not inherit the old cu121 pins)

The RTX 5070 Ti is **Blackwell (sm_120)**. PyTorch built for CUDA 12.1 (cu121) — which the *old*
project's `requirements.txt` specified — **does not support sm_120** and will fail or silently fall
back. torch here MUST come from the **cu128** wheel index:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

After install, verify:

```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # -> RTX 5070 Ti (12, 0)
```

### Perceptual-loss extras

`asteroid` and `torch_pesq` (used by the legacy training loss) can lag the newest torch. The setup
script installs them **best-effort** and continues if they conflict. They are only needed by the
inherited loss, which the new architecture may replace — do not block the environment on them.

## Dataset — LRS3

| Item | Value |
|---|---|
| Root | `D:\DataSet\LRS3` |
| Splits | `train/`, `dev/`, `eval/`, `test/` |
| Audio | `<split>/scenes/<scene_id>_{mixed,target,interferer}.wav` (16 kHz) |
| Video | `<split>/96/<scene_id>.npy` — preprocessed 96×96 **uint8** grayscale, shape `[T, 96, 96]` |
| Metadata | `scenes.train.json`, `scenes.dev.json` (scene → interferer type, etc.) |
| Prep scripts | live with the dataset: `MixWithSNR.py`, `Separate_video_audio.py`, `preprocess_video_96.py`, ... |

The migrated data pipeline (`src/avse/data/`) reads this layout directly; set `data.root_dir:
D:/DataSet/LRS3` in the config.

## Open environment item

- **Teacher checkpoint** (the 0.37 M reference model, for the optional distillation route) lived on
  the original `E:\` machine (`.../new_structure_No10_ND/final_model.ckpt`) and is **not yet located
  on this machine.** Needed only when/if Phase 2 pursues distillation — ask the owner then.

## Phase-3 toolchain (not needed until finalists are chosen)

| Tool | Version | Use |
|---|---|---|
| Vitis HLS | **2022.2** (installed at `D:\Xilinx\Vitis_HLS\2022.2`; `bin/vitis_hls.bat`) | C-synth of the HLS IP(s) |
| Vivado | **2022.2** (`D:\Xilinx\Vivado\2022.2`; `bin/vivado.bat`) | synth + place-and-route + bitstream |
| Board | Real Digital RFSoC 4x2 (ZU48DR, `xczu48dr-ffvg1517-2-e`) | `ssh xilinx@172.26.206.133` (pw `xilinx`, DHCP — verify IP). PYNQ env: `source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh`; PL ops need `sudo`. |
