#!/usr/bin/env bash
# download DECIMER model (download 285MB  HandDrawn model)
set -euo pipefail
python << 'PY'
import importlib.util
import sys
from pathlib import Path

import pystow

decimer_dir = None
for p in sys.path:
    cand = Path(p) / "DECIMER" / "utils.py"
    if cand.is_file():
        decimer_dir = cand.parent
        break
if decimer_dir is None:
    raise SystemExit("please: pip install decimer")

spec = importlib.util.spec_from_file_location("decimer_utils", decimer_dir / "utils.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

home = pystow.join("DECIMER-V2")
url = "https://zenodo.org/record/8300489/files/models.zip"
print("Downloading DECIMER main model to", home)
mod.ensure_models(home, {"DECIMER": url})
pb = Path(home) / "DECIMER_model" / "saved_model.pb"
print("Done:", pb, "size", pb.stat().st_size if pb.is_file() else 0)
PY
