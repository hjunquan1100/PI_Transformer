#!/usr/bin/env bash
# Start the backend API and frontend Vite server.
# By default, services are launched in the background.
# For foreground mode with Ctrl+C cleanup: PI_TG_FOREGROUND=1 bash scripts/start-web.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_LOG="${PI_TG_BACKEND_LOG:-/tmp/pi-tg-backend.log}"
FRONTEND_LOG="${PI_TG_FRONTEND_LOG:-/tmp/pi-tg-frontend.log}"
BACKEND_PID_FILE="${PI_TG_BACKEND_PID:-/tmp/pi-tg-backend.pid}"
FRONTEND_PID_FILE="${PI_TG_FRONTEND_PID:-/tmp/pi-tg-frontend.pid}"

stop_pidfile() {
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[start-web] Stopping ${label} pid=$pid"
      kill "$pid" 2>/dev/null || true
      pkill -P "$pid" 2>/dev/null || true
      sleep 0.5
      kill -9 "$pid" 2>/dev/null || true
      pkill -9 -P "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

cleanup_all() {
  stop_pidfile "$FRONTEND_PID_FILE" "frontend"
  stop_pidfile "$BACKEND_PID_FILE" "backend"
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  pkill -f "node_modules/vite/bin/vite.js --host 0.0.0.0 --port 5173" 2>/dev/null || true
}

if [[ "${1:-}" == "stop" ]]; then
  cleanup_all
  echo "[start-web] Stopped"
  exit 0
fi

echo "[start-web] Cleaning old processes..."
cleanup_all || true
sleep 1

echo "[start-web] Starting backend -> http://0.0.0.0:8000  (log: $BACKEND_LOG)"
nohup bash "$SCRIPT_DIR/start-backend.sh" >"$BACKEND_LOG" 2>&1 &
echo $! >"$BACKEND_PID_FILE"

ok=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done
if [[ "$ok" -ne 1 ]]; then
  echo "[start-web] Backend startup timed out. See: $BACKEND_LOG" >&2
  tail -n 50 "$BACKEND_LOG" >&2 || true
  exit 1
fi
echo "[start-web] Backend is ready"

echo "[start-web] Starting frontend -> http://0.0.0.0:5173  (log: $FRONTEND_LOG)"
if [[ "${PI_TG_FOREGROUND:-0}" == "1" ]]; then
  trap cleanup_all EXIT INT TERM
  echo "[start-web] Foreground mode: Ctrl+C stops both services"
  echo "[start-web] Open: http://localhost:5173"
  bash "$SCRIPT_DIR/start-frontend.sh"
else
  nohup bash "$SCRIPT_DIR/start-frontend.sh" >"$FRONTEND_LOG" 2>&1 &
  echo $! >"$FRONTEND_PID_FILE"
  ok=0
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:5173/" >/dev/null 2>&1 \
      && curl -sf "http://127.0.0.1:5173/api/health" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 1
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[start-web] Frontend startup timed out. See: $FRONTEND_LOG" >&2
    tail -n 50 "$FRONTEND_LOG" >&2 || true
    exit 1
  fi
  echo "[start-web] Frontend is ready (background mode)"
  echo "[start-web] Open: http://localhost:5173"
  echo "[start-web] Prediction: http://localhost:5173/   Inverse: http://localhost:5173/inverse"
  echo "[start-web] Stop: bash $SCRIPT_DIR/start-web.sh stop"
fi
