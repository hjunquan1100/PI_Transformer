"""PI inverse design: conditional generation from target Tg."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import (
    INVERSE_MAX_ABS_ERROR,
    INVERSE_N_SAMPLES,
    INVERSE_TG_MAX,
    INVERSE_TG_MIN,
    INVERSE_TOP_K,
    PI_INVERSE_CKPT,
    PI_INVERSE_FWD,
    PI_INVERSE_SRC,
)


class InverseGenerateError(ValueError):
    """Invalid input or no structures passed PI validation."""


@dataclass
class _InverseRuntime:
    model: Any
    vocab: dict
    inv_vocab: dict
    stats: dict
    device: str
    generate_batch: Any
    predict_tg: Any


def _load_module_from_src(module_name: str, filename: str):
    path = PI_INVERSE_SRC / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing inverse-design module: {path}")
    src = str(PI_INVERSE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def get_inverse_runtime() -> _InverseRuntime:
    if not PI_INVERSE_CKPT.is_file():
        raise FileNotFoundError(f"Inverse generator checkpoint does not exist: {PI_INVERSE_CKPT}")

    gen_mod = _load_module_from_src("pi_generate", "generate.py")
    fwd_mod = _load_module_from_src("pi_forward_model", "forward_model.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, inv_vocab, stats = gen_mod.load_model(str(PI_INVERSE_CKPT), device=device)

    return _InverseRuntime(
        model=model,
        vocab=vocab,
        inv_vocab=inv_vocab,
        stats=stats,
        device=device,
        generate_batch=gen_mod.generate_batch,
        predict_tg=fwd_mod.predict_tg,
    )


def inverse_model_loaded() -> bool:
    """Checkpoint files present (does not eagerly load weights)."""
    return PI_INVERSE_CKPT.is_file() and PI_INVERSE_FWD.is_file()


def normalize_tg_target(tg_target_c: float, stats: dict | None = None) -> float:
    tg = round(float(tg_target_c), 1)
    tg_min = float((stats or {}).get("tg_min", INVERSE_TG_MIN))
    tg_max = float((stats or {}).get("tg_max", INVERSE_TG_MAX))
    if tg < tg_min or tg > tg_max:
        raise InverseGenerateError(
            f"Target Tg must be between {tg_min} and {tg_max} deg C; received {tg}."
        )
    return tg


def _score_passed(passed: list[dict], tg_target: float, predict_tg_fn) -> list[dict]:
    rows = [r for r in passed if r.get("smiles")]
    if not rows:
        return []

    smiles_list = [r["smiles"] for r in rows]
    preds = predict_tg_fn(smiles_list, model_path=str(PI_INVERSE_FWD))
    scored = []
    for r, pred in zip(rows, preds):
        if not r.get("smiles"):
            continue
        if pred is None or (isinstance(pred, float) and np.isnan(pred)):
            continue
        pred_f = round(float(pred), 1)
        err = round(pred_f - tg_target, 1)
        scored.append(
            {
                "id": r.get("id", ""),
                "smiles": r["smiles"],
                "tg_target_c": tg_target,
                "pred_tg_c": pred_f,
                "tg_error_c": err,
                "valid": True,
            }
        )
    scored.sort(key=lambda x: abs(x["tg_error_c"]))
    return scored


def run_inverse_generate(tg_target_c: float) -> dict:
    rt = get_inverse_runtime()
    tg_target = normalize_tg_target(tg_target_c, rt.stats)

    output = rt.generate_batch(
        model=rt.model,
        tg_target=tg_target,
        vocab=rt.vocab,
        inv_vocab=rt.inv_vocab,
        stats=rt.stats,
        n_samples=INVERSE_N_SAMPLES,
        temperature=0.8,
        top_p=0.9,
        device=rt.device,
        validate=True,
        strict=True,
        verbose=False,
    )

    passed = output.get("passed_results") or []
    valid_count = int(output.get("valid_count", 0))
    passed_count = len(passed)

    if passed_count == 0:
        raise InverseGenerateError(
            f"Generated {INVERSE_N_SAMPLES} candidates, but no structures passed PI plausibility validation. "
            "Adjust the target Tg and retry."
        )

    scored = _score_passed(passed, tg_target, rt.predict_tg)
    if not scored:
        raise InverseGenerateError(
            "Some structures passed PI validation, but the forward Tg model could not score them. "
            "Check the forward-model path."
        )

    in_band = [x for x in scored if abs(x["tg_error_c"]) <= INVERSE_MAX_ABS_ERROR]
    if in_band:
        pool = in_band
    else:
        # Return closest candidates when none fall in the error band.
        pool = scored

    recommended = []
    others = []
    for i, item in enumerate(pool):
        row = {**item, "rank": i + 1}
        if i < INVERSE_TOP_K:
            recommended.append(row)
        else:
            others.append(row)

    # Put remaining scored results into the expandable list.
    rec_ids = {r["id"] for r in recommended}
    for item in scored:
        if item["id"] in rec_ids:
            continue
        if any(o["id"] == item["id"] for o in others):
            continue
        others.append({**item, "rank": len(recommended) + len(others) + 1})

    return {
        "tg_target_c": tg_target,
        "n_generated": INVERSE_N_SAMPLES,
        "valid_count": valid_count,
        "passed_count": passed_count,
        "recommended": recommended,
        "others": others,
    }
