#!/usr/bin/env bash
# Create the pi conda environment, install demo/Web dependencies, and download the DECIMER main model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${PI_CONDA_ENV:-pi}"
PYTHON_VERSION="${PI_PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    echo "[setup_env] Error: conda was not found" >&2
    exit 1
  fi
fi

echo "============================================================"
echo "[setup_env] Repository root: $ROOT"
echo "[setup_env] conda:      $CONDA_BIN"
echo "[setup_env] Environment:     $ENV_NAME (Python $PYTHON_VERSION)"
echo "============================================================"

# shellcheck disable=SC1091
eval "$("$CONDA_BIN" shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup_env] Environment already exists: $ENV_NAME; skipping create"
else
  echo "[setup_env] Creating environment $ENV_NAME ..."
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"
PYTHON="$(command -v python)"
PIP="$(command -v pip)"
echo "[setup_env] Using Python: $PYTHON"

echo "[setup_env] Upgrading pip / setuptools / wheel ..."
"$PIP" install -U pip setuptools wheel

echo "[setup_env] Installing PyTorch (using cu121 wheels when CUDA is available) ..."
if command -v nvidia-smi >/dev/null 2>&1; then
  "$PIP" install "torch>=2.0.0,<3.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" \
    --index-url https://download.pytorch.org/whl/cu121 \
    || "$PIP" install "torch>=2.0.0,<3.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
else
  "$PIP" install "torch>=2.0.0,<3.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
fi

echo "[setup_env] Installing pi_forward_prediction dependencies ..."
"$PIP" install -r "$ROOT/pi_forward_prediction/requirements.txt"

echo "[setup_env] Installing pi_inverse_design dependencies ..."
"$PIP" install -r "$ROOT/pi_inverse_design/requirements.txt"

echo "[setup_env] Installing Web backend and demo dependencies ..."
"$PIP" install -r "$ROOT/web/prediction/backend/requirements.txt"
# TensorFlow is used by DECIMER; image_parser defaults to CPU.
"$PIP" install "tensorflow>=2.12,<2.16" pystow Pillow

echo "[setup_env] Trying to install system libraries required by RDKit drawing and DECIMER ..."
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq libxrender1 libgl1-mesa-glx libglib2.0-0 || true
  elif [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -qq || true
    apt-get install -y -qq libxrender1 libgl1-mesa-glx libglib2.0-0 || true
  else
    echo "  No root permission. Install manually: sudo apt-get install -y libxrender1 libgl1-mesa-glx libglib2.0-0"
  fi
else
  echo "  Non-apt system. Install libXrender and libGL manually."
fi

echo "[setup_env] Downloading DECIMER main model ..."
bash "$ROOT/web/prediction/scripts/setup_decimer_model.sh"

echo "[setup_env] Smoke check ..."
"$PYTHON" - <<'PY'
import sys
from pathlib import Path

errors = []
for mod in ("torch", "rdkit", "selfies", "transformers", "sklearn", "fastapi", "joblib"):
    try:
        __import__(mod)
        print(f"  OK  import {mod}")
    except Exception as e:
        errors.append(f"{mod}: {e}")
        print(f"  FAIL import {mod}: {e}")

pb = Path.home() / ".data" / "DECIMER-V2" / "DECIMER_model" / "saved_model.pb"
if pb.is_file() and pb.stat().st_size > 1_000_000:
    print(f"  OK  DECIMER model: {pb} ({pb.stat().st_size} bytes)")
else:
    errors.append(f"DECIMER model missing: {pb}")
    print(f"  FAIL DECIMER model: {pb}")

if errors:
    print("Smoke check failed:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("Smoke check passed.")
PY

echo "============================================================"
echo "[setup_env] Complete. Next steps:"
echo "  conda activate $ENV_NAME"
echo "  bash $ROOT/scripts/run_demo.sh --tg_target 300 --pause"
echo "  bash $ROOT/web/prediction/scripts/start-backend.sh"
echo "  bash $ROOT/web/prediction/scripts/start-frontend.sh"
echo "============================================================"
