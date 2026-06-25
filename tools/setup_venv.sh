#!/usr/bin/env bash
# ============================================================================
# setup_venv.sh — create the project virtual environment.
#
#   Python : 3.11  (D:/.. tools + reference-inference compatibility)
#   GPU    : RTX 5070 Ti (Blackwell, sm_120) -> torch MUST be cu128.
#
# Idempotent-ish: re-running re-uses the existing .venv. Logs to tools/setup_venv.log.
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/tools/setup_venv.log"
exec > >(tee "$LOG") 2>&1

PY311="/c/Users/dongk/AppData/Local/Programs/Python/Python311/python.exe"
echo "== [1/6] python =="
"$PY311" --version

echo "== [2/6] create venv =="
if [ ! -d "$ROOT/.venv" ]; then
  "$PY311" -m venv "$ROOT/.venv"
fi
VPY="$ROOT/.venv/Scripts/python.exe"
"$VPY" --version

echo "== [3/6] upgrade pip =="
"$VPY" -m pip install --upgrade pip wheel setuptools

echo "== [4/6] torch (cu128, Blackwell) =="
"$VPY" -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "== [4b] verify CUDA =="
"$VPY" - <<'PYEOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability sm_%d%d" % torch.cuda.get_device_capability(0))
PYEOF

echo "== [5/6] core requirements =="
"$VPY" -m pip install -r "$ROOT/requirements.txt"

echo "== [5b] perceptual-loss extras (best-effort; may skip on conflict) =="
"$VPY" -m pip install asteroid torch_pesq || echo "WARN: asteroid/torch_pesq skipped (resolve later; loss may be redesigned)"

echo "== [6/6] editable install of src/avse =="
"$VPY" -m pip install -e "$ROOT"

echo "== DONE =="
