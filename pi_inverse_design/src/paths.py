"""Project-relative default paths for PI inverse design scripts."""
from __future__ import annotations

from pathlib import Path

PI_INVERSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PI_INVERSE_ROOT.parent

DEFAULT_RAW_DATA = PI_INVERSE_ROOT / "data" / "raw" / "data" / "data.xlsx"
DEFAULT_PROCESSED_DIR = PI_INVERSE_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT_DIR = PI_INVERSE_ROOT / "checkpoints"
DEFAULT_RESULTS_DIR = PI_INVERSE_ROOT / "results"

DEFAULT_GENERATOR_CKPT = (
    DEFAULT_CHECKPOINT_DIR
    / "reverse_20260318_232126"
    / "best_model_valloss_0.9011.pt"
)
DEFAULT_FORWARD_MODEL = (
    DEFAULT_CHECKPOINT_DIR
    / "forward_20260318_232230"
    / "forward_model_best_r2_0.8535.pkl"
)
DEFAULT_FORWARD_MODEL_OUT = DEFAULT_CHECKPOINT_DIR / "forward_model.pkl"
