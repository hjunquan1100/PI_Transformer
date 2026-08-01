#!/usr/bin/env python3
"""Validate that the packaged UniPolymer runtime artifacts are present."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "MANIFEST.md",
    "requirements.txt",
    ".gitattributes",
    "pi_forward_prediction/Downstream.py",
    "pi_forward_prediction/PolymerSmilesTokenization.py",
    "pi_forward_prediction/dataset.py",
    "pi_forward_prediction/prep_PI_Tg_descriptors.py",
    "pi_forward_prediction/scripts/extend_pi_tg_pool_features.py",
    "pi_forward_prediction/experiments/config/p26_runs/kfold_m128_head15.yaml",
    "pi_forward_prediction/data/PI_Tg_10066.csv",
    "pi_forward_prediction/data/PI_Tg_10066_with_desc_p1_m128.csv",
    "pi_forward_prediction/data/PI_Tg_10066_holdout80_p1_m128.csv",
    "pi_forward_prediction/data/PI_Tg_with_desc_train_pool_p1_m128.csv",
    "pi_forward_prediction/data/PI_Tg_with_desc_holdout80_p1_m128.csv",
    "pi_forward_prediction/data/vocab/vocab_sup_PE_I.csv",
    "pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/config.json",
    "pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/model.safetensors",
    "pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/vocab.json",
    "pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/merges.txt",
    "pi_forward_prediction/ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt",
    "pi_inverse_design/src/paths.py",
    "pi_inverse_design/src/generate.py",
    "pi_inverse_design/src/model.py",
    "pi_inverse_design/src/validator.py",
    "pi_inverse_design/src/forward_model.py",
    "pi_inverse_design/data/raw/data/PI_Tg_10066.csv",
    "pi_inverse_design/data/raw/data/data.xlsx",
    "pi_inverse_design/data/processed/train_augmented.pkl",
    "pi_inverse_design/data/processed/val.pkl",
    "pi_inverse_design/data/processed/vocab.json",
    "pi_inverse_design/data/processed/stats.json",
    "pi_inverse_design/checkpoints/reverse_20260318_232126/best_model_valloss_0.9011.pt",
    "pi_inverse_design/checkpoints/forward_20260318_232230/forward_model_best_r2_0.8535.pkl",
    "web/prediction/backend/app/main.py",
    "web/prediction/backend/app/inference.py",
    "web/prediction/backend/app/inverse_service.py",
    "web/prediction/backend/requirements.txt",
    "web/prediction/frontend/package.json",
    "web/prediction/frontend/package-lock.json",
    "web/prediction/frontend/src/App.vue",
    "web/prediction/scripts/start-web.sh",
]


def main() -> int:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        print("Missing required files:")
        for rel in missing:
            print(f"  - {rel}")
        return 1

    large = [
        "pi_forward_prediction/ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt",
        "pi_forward_prediction/ckpt/pretrain.pt/pytorch_model.bin",
        "pi_forward_prediction/ckpt/experiments/full/contrastive_roberta/model.safetensors",
    ]
    print("UniPolymer package check passed.")
    for rel in large:
        path = ROOT / rel
        if path.is_file():
            print(f"  {rel}: {path.stat().st_size / (1024 ** 2):.1f} MiB")
    print("Reminder: use Git LFS for large model artifacts before pushing to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
