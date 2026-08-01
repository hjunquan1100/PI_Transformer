#!/usr/bin/env bash
# Activate the pi environment and run the three-step terminal demo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${PI_CONDA_ENV:-pi}"
CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi

# shellcheck disable=SC1091
eval "$("$CONDA_BIN" shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[run_demo] Conda environment '$ENV_NAME' was not found." >&2
  echo "Please run: bash $ROOT/scripts/setup_env.sh" >&2
  exit 1
fi

conda activate "$ENV_NAME"

export PYTHONPATH="$ROOT/web/prediction/backend:$ROOT/pi_forward_prediction${PYTHONPATH:+:$PYTHONPATH}"
export PI_FORWARD_ROOT="${PI_FORWARD_ROOT:-$ROOT/pi_forward_prediction}"
export PI_INVERSE_ROOT="${PI_INVERSE_ROOT:-$ROOT/pi_inverse_design}"
# demo_three_step.py loads inverse src on demand to avoid shadowing forward dataset.py/model.py.

echo "[run_demo] Python: $(command -v python)"
echo "[run_demo] PYTHONPATH is set"
echo "[run_demo] Running: python $ROOT/scripts/demo_three_step.py $*"
echo

exec python "$ROOT/scripts/demo_three_step.py" "$@"
