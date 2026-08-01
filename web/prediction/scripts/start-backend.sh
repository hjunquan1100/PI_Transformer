#!/usr/bin/env bash
# Start the FastAPI backend and auto-select a conda environment with torch and rdkit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=""
# Prefer the pi demo/Web environment, then fall back to other common environments.
for env in pi slai-dara-gpu tomvae; do
  CAND="/root/anaconda3/envs/${env}/bin/python"
  if [[ -x "$CAND" ]] && "$CAND" -c "import torch, rdkit" 2>/dev/null; then
    PYTHON="$CAND"
    echo "[start-backend] Using conda environment: $env"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  if python -c "import torch, rdkit" 2>/dev/null; then
    PYTHON="python"
    echo "[start-backend] Using current python"
  else
    echo "[start-backend] Error: no torch+rdkit environment found (tried pi / slai-dara-gpu / tomvae)" >&2
    echo "[start-backend] Please run: bash scripts/setup_env.sh" >&2
    exit 1
  fi
fi

exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
