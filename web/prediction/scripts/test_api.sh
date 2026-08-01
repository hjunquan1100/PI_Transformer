#!/usr/bin/env bash
#  API smoketest (backend :8000 run)
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== health =="
curl -sf "$BASE/api/health" | python3 -m json.tool

SMI=$(python3 -c "import pandas as pd; print(pd.read_csv('$(cd "$(dirname "$0")/../../.." && pwd)/pi_forward_prediction/data/PI_Tg_10066_with_desc_p1_m128.csv', nrows=1)['smiles'].iloc[0])")

echo "== predict/smiles =="
curl -sf -X POST "$BASE/api/predict/smiles" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json,sys; print(json.dumps({'smiles': sys.argv[1]}))" "$SMI")" | python3 -m json.tool

echo "== structure/svg =="
code=$(curl -s -o /dev/null -w "%{http_code}" -G "$BASE/api/structure/svg" --data-urlencode "smiles=$SMI")
echo "HTTP $code"

echo "== invalid smiles (expect 422) =="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/predict/smiles" \
  -H 'Content-Type: application/json' -d '{"smiles":"invalid@@@"}')
echo "HTTP $code"

echo "OK"
