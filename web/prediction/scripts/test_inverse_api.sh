#!/usr/bin/env bash
# inversegenerate API smoketest (backend :8000 run)
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== health (inverse fields) =="
curl -sf "$BASE/api/health" | python3 -m json.tool | grep -E 'inverse|model_loaded' || true

echo "== inverse generate tg=300.0 (may take 1-5 min) =="
RESP=$(curl -sf -X POST "$BASE/api/inverse/generate" \
  -H 'Content-Type: application/json' \
  -d '{"tg_target_c": 300.0}')
echo "$RESP" | python3 -m json.tool | head -40

TMP=$(mktemp)
echo "$RESP" > "$TMP"
python3 -c "
import json
with open('$TMP') as f:
    d = json.load(f)
assert d['tg_target_c'] == 300.0
assert d['n_generated'] == 15
assert 1 <= len(d['recommended']) <= 5
assert d['recommended'][0]['smiles']
assert 'pred_tg_c' in d['recommended'][0]
print('inverse generate OK, recommended=', len(d['recommended']))
"
rm -f "$TMP"

echo "== invalid tg (expect 422) =="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/inverse/generate" \
  -H 'Content-Type: application/json' -d '{"tg_target_c": 1000.0}')
echo "HTTP $code"
test "$code" = "422"

echo "ALL OK"
