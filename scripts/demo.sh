#!/usr/bin/env bash
# Short entry point equivalent to:
#   bash scripts/run_demo.sh --tg_target 300 --n_samples 15 --max_abs_error 25 --top_k 5
# Additional arguments are appended, for example: bash scripts/demo.sh --pause
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_demo.sh" \
  --tg_target 300 \
  --n_samples 15 \
  --max_abs_error 25 \
  --top_k 5 \
  "$@"
