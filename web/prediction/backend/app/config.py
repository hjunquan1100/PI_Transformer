"""Paths and settings for PI Tg prediction API."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ROOT = BACKEND_ROOT.parent
REPO_ROOT = PREDICTION_ROOT.parent.parent

PI_FORWARD_ROOT = Path(
    os.environ.get("PI_FORWARD_ROOT", str(REPO_ROOT / "pi_forward_prediction"))
).resolve()

PI_TG_CONFIG = Path(
    os.environ.get(
        "PI_TG_CONFIG",
        str(
            PI_FORWARD_ROOT
            / "experiments/config/p26_runs/kfold_m128_head15.yaml"
        ),
    )
).resolve()

PI_TG_CKPT = Path(
    os.environ.get(
        "PI_TG_CKPT",
        str(
            PI_FORWARD_ROOT
            / "ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt"
        ),
    )
).resolve()

ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
SCALERS_PATH = ARTIFACTS_DIR / "scalers.joblib"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}

PI_INVERSE_ROOT = Path(
    os.environ.get("PI_INVERSE_ROOT", str(REPO_ROOT / "pi_inverse_design"))
).resolve()

PI_INVERSE_SRC = PI_INVERSE_ROOT / "src"

PI_INVERSE_CKPT = Path(
    os.environ.get(
        "PI_INVERSE_CKPT",
        str(
            PI_INVERSE_ROOT
            / "checkpoints/reverse_20260318_232126/best_model_valloss_0.9011.pt"
        ),
    )
).resolve()

PI_INVERSE_FWD = Path(
    os.environ.get(
        "PI_INVERSE_FWD",
        str(
            PI_INVERSE_ROOT
            / "checkpoints/forward_20260318_232230/forward_model_best_r2_0.8535.pkl"
        ),
    )
).resolve()

INVERSE_N_SAMPLES = 15
INVERSE_TOP_K = 5
INVERSE_MAX_ABS_ERROR = 25.0  # Recommended maximum |predicted Tg - target Tg|.
INVERSE_TG_MIN = 19.0
INVERSE_TG_MAX = 460.0
