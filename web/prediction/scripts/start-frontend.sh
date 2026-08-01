#!/usr/bin/env bash
# On NFS/shared filesystems, node_modules can lack executable permissions.
# Install and run frontend dependencies under /tmp by default.
# package.json overrides rollup with @rollup/wasm-node for older GLIBC systems.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="$ROOT/frontend"
TMP_FE="${PI_TG_FE_TMP:-/tmp/pi-tg-fe-dev}"

esbuild_ok() {
  local bin="$1/node_modules/@esbuild/linux-x64/bin/esbuild"
  [[ -x "$bin" ]]
}

rollup_ok() {
  # With the wasm override, requiring rollup is enough.
  (cd "$1" && node -e "require('rollup');" >/dev/null 2>&1) \
    || (cd "$1" && node -e "import('rollup').then(()=>{}).catch(e=>{console.error(e);process.exit(1)})" >/dev/null 2>&1)
}

sync_sources() {
  mkdir -p "$TMP_FE"
  for f in package.json package-lock.json vite.config.ts tsconfig.json tsconfig.node.json index.html; do
    [[ -f "$FRONTEND/$f" ]] && cp -f "$FRONTEND/$f" "$TMP_FE/$f"
  done
  # Always sync the latest package.json, including rollup wasm overrides.
  cp -f "$FRONTEND/package.json" "$TMP_FE/package.json"
  rm -rf "$TMP_FE/src"
  cp -a "$FRONTEND/src" "$TMP_FE/src"
}

echo "[start-frontend] Node: $(command -v node) ($(node -v))"
echo "[start-frontend] directory: $TMP_FE"

if [[ "${PI_TG_USE_LOCAL:-}" == "1" ]] && esbuild_ok "$FRONTEND"; then
  echo "[start-frontend] Using project node_modules (PI_TG_USE_LOCAL=1)"
  cd "$FRONTEND"
  exec npm run dev -- --host 0.0.0.0 --port 5173
fi

sync_sources

need_install=0
if ! esbuild_ok "$TMP_FE"; then
  need_install=1
elif ! rollup_ok "$TMP_FE"; then
  echo "[start-frontend] Rollup is unavailable; reinstalling dependencies..."
  need_install=1
fi

if [[ "$need_install" -eq 1 ]]; then
  echo "[start-frontend] Installing npm dependencies under /tmp (first run may take 1-2 minutes)..."
  rm -rf "$TMP_FE/node_modules" "$TMP_FE/package-lock.json"
  (cd "$TMP_FE" && npm install --no-audit --no-fund)
fi

if ! esbuild_ok "$TMP_FE"; then
  echo "[start-frontend] Error: esbuild is still not executable. Check Node.js and npm." >&2
  exit 1
fi

if ! rollup_ok "$TMP_FE"; then
  echo "[start-frontend] Error: Rollup still cannot load. Check package.json rollup overrides." >&2
  exit 1
fi

cd "$TMP_FE"
echo "[start-frontend] Starting Vite -> http://0.0.0.0:5173"
exec npm run dev -- --host 0.0.0.0 --port 5173
