#!/usr/bin/env python3
"""
Three-step terminal demo.

  Step 1: target Tg -> inverse generation -> structure PNG
  Step 2: structure PNG -> DECIMER -> TransPolymer Tg prediction
  Step 3: generated SMILES -> L1/L2/L3 physical filters -> final candidates
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Runtime paths.
REPO_ROOT = Path(__file__).resolve().parents[1]
INVERSE_SRC = REPO_ROOT / "pi_inverse_design" / "src"
BACKEND_ROOT = REPO_ROOT / "web" / "prediction" / "backend"
FORWARD_ROOT = REPO_ROOT / "pi_forward_prediction"

# Put the web backend and forward predictor on sys.path first. The inverse
# stack is loaded on demand to avoid dataset.py/model.py name collisions.
for p in (str(FORWARD_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PI_FORWARD_ROOT", str(FORWARD_ROOT))
os.environ.setdefault(
    "PI_INVERSE_ROOT", str(REPO_ROOT / "pi_inverse_design")
)


def _load_inverse_stack():
    """
    Load inverse-design modules in isolation.

    Returns (generate_mod, validator_mod, forward_mod).
    """
    import importlib.util

    src = str(INVERSE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)

    def _load(short: str, filename: str):
        path = INVERSE_SRC / filename
        spec = importlib.util.spec_from_file_location(f"pi_demo_{short}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[short] = mod
        sys.modules[f"pi_demo_{short}"] = mod
        spec.loader.exec_module(mod)
        return mod

    # model and validator must be available before generate.py is imported.
    _load("model", "model.py")
    validator_mod = _load("validator", "validator.py")
    generate_mod = _load("generate", "generate.py")
    forward_mod = _load("forward_model", "forward_model.py")
    return generate_mod, validator_mod, forward_mod


def _unload_inverse_stack() -> None:
    """Unload inverse modules so forward modules with the same names remain isolated."""
    src = str(INVERSE_SRC)
    if src in sys.path:
        sys.path.remove(src)
    for victim in ("dataset", "model", "validator", "generate", "forward_model"):
        m = sys.modules.get(victim)
        if m is not None and src in str(getattr(m, "__file__", "") or ""):
            del sys.modules[victim]
    for key in list(sys.modules):
        if key.startswith("pi_demo_"):
            del sys.modules[key]

DEFAULT_INVERSE_CKPT = (
    REPO_ROOT
    / "pi_inverse_design"
    / "checkpoints"
    / "reverse_20260318_232126"
    / "best_model_valloss_0.9011.pt"
)
DEFAULT_INVERSE_FWD = (
    REPO_ROOT
    / "pi_inverse_design"
    / "checkpoints"
    / "forward_20260318_232230"
    / "forward_model_best_r2_0.8535.pkl"
)


BANNER_W = 72


def banner(title: str) -> None:
    line = "=" * BANNER_W
    print()
    print(line)
    print(title.center(BANNER_W))
    print(line)
    print()


def subbanner(text: str) -> None:
    print("-" * BANNER_W)
    print(text)
    print("-" * BANNER_W)


def maybe_pause(enabled: bool, step_label: str) -> None:
    if not enabled:
        return
    print()
    try:
        input(f">>> [{step_label}] Press Enter to continue ... ")
    except EOFError:
        print("(stdin is not interactive; continuing)")


def short_smiles(s: str, n: int = 64) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def step1_generate(
    tg_target: float,
    n_samples: int,
    ckpt: Path,
    out_dir: Path,
    temperature: float,
    top_p: float,
) -> tuple[list[dict[str, Any]], Path]:
    banner("STEP 1/3  Generate Candidate Structures")
    print("Goal: generate polyimide repeat-unit candidates from a target Tg.")
    print("Method: PI inverse-design generator (SELFIES + top-p sampling).")
    print(f"input:   target Tg = {tg_target:.1f}  deg C, candidate = {n_samples}")
    print(f"output:  SMILES rows + PNG structure images -> {out_dir / 'structures'}")
    print(f"checkpoint:   {ckpt}")
    print()

    import torch

    gen_mod, _, _ = _load_inverse_stack()
    try:
        load_model = gen_mod.load_model
        generate_batch = gen_mod.generate_batch
        draw_structure_images = gen_mod.draw_structure_images
        save_results_csv = gen_mod.save_results_csv

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Step1] Device: {device}")
        t0 = time.time()
        model, vocab, inv_vocab, stats = load_model(str(ckpt), device=device)
        print(f"[Step1] Model loaded in {time.time() - t0:.1f}s")

        # Physical plausibility filtering is run in Step 3.
        t1 = time.time()
        output = generate_batch(
            model=model,
            tg_target=tg_target,
            vocab=vocab,
            inv_vocab=inv_vocab,
            stats=stats,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            device=device,
            validate=False,
            strict=True,
            verbose=True,
        )
        print(f"[Step1] Generation finished in {time.time() - t1:.1f}s")

        all_results = output["all_results"]
        structures_dir = out_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)

        save_results_csv(str(out_dir / "generated_all.csv"), all_results)
        n_img = draw_structure_images(all_results, str(structures_dir), verbose=True)
    finally:
        _unload_inverse_stack()

    subbanner(f"Step 1 result: RDKit parsed {output['valid_count']}/{n_samples}; {n_img} images")
    for row in all_results:
        mol_id = row.get("id", "?")
        smi = row.get("smiles") or ""
        png = structures_dir / f"{mol_id}.png"
        if smi and png.is_file():
            print(f"  {mol_id}  PNG={png.name}")
            print(f"           SMILES={short_smiles(smi)}")
        elif smi:
            print(f"  {mol_id}  PNG missing; SMILES={short_smiles(smi)}")
        else:
            print(f"  {mol_id}  FAIL invalid or empty SMILES")

    return all_results, structures_dir


def step2_image_predict(
    all_results: list[dict[str, Any]],
    structures_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Return mol_id -> prediction."""
    banner("STEP 2/3  Structure Image to Tg Prediction")
    print("input:   structures/*.png")
    print("output:  DECIMER-recognized SMILES and predicted Tg (deg C)")
    print()

    from app.descriptors import SmilesDescriptorError
    from app.image_parser import ImageParseError, decimer_available, image_bytes_to_smiles
    from app.inference import get_predictor

    if not decimer_available():
        print("[Step2] DECIMER is not ready. Run:")
        print("  bash scripts/setup_env.sh")
        print("  or: bash web/prediction/scripts/setup_decimer_model.sh")
        raise SystemExit(2)

    print("[Step2] Loading TransPolymer Tg predictor ...")
    t0 = time.time()
    predictor = get_predictor()
    print(f"[Step2] Predictor ready (device={predictor.device}, {time.time() - t0:.1f}s)")
    print("[Step2] DECIMER may take a moment while TensorFlow initializes.")
    print()

    preds: dict[str, dict[str, Any]] = {}
    for row in all_results:
        mol_id = row.get("id", "?")
        true_smi = row.get("smiles") or ""
        png = structures_dir / f"{mol_id}.png"
        rec: dict[str, Any] = {
            "id": mol_id,
            "true_smiles": true_smi,
            "image_path": str(png) if png.is_file() else "",
            "parsed_smiles": None,
            "pred_tg_c": None,
            "tg_source": None,
            "error": None,
        }

        if not true_smi:
            rec["error"] = "Step 1 did not produce a valid SMILES for image prediction"
            print(f"  [{mol_id}]  - {rec['error']}")
            preds[mol_id] = rec
            continue
        if not png.is_file():
            rec["error"] = f"Structure image does not exist: {png}"
            print(f"  [{mol_id}]  - {rec['error']}")
            preds[mol_id] = rec
            continue

        print(f"  [{mol_id}] image: {png}")
        try:
            img_bytes = png.read_bytes()
            parsed = image_bytes_to_smiles(img_bytes, suffix=".png")
            rec["parsed_smiles"] = parsed
            print(f"           DECIMER -> {short_smiles(parsed)}")
            tg = predictor.predict_tg(parsed)
            rec["pred_tg_c"] = round(float(tg), 2)
            rec["tg_source"] = "structure image DECIMER"
            print(f"           TransPolymer -> predicted Tg = {rec['pred_tg_c']:.2f} deg C")
        except (ImageParseError, SmilesDescriptorError) as exc:
            rec["error"] = str(exc)
            print(f"           FAIL: {exc}")
        except Exception as exc:
            rec["error"] = f"Prediction failed: {exc}"
            print(f"           FAIL: {exc}")

        preds[mol_id] = rec
        print()

    ok = sum(1 for r in preds.values() if r.get("pred_tg_c") is not None)
    subbanner(f"Step 2 complete: {ok}/{len(preds)} image-based predictions succeeded")
    return preds


def step3_filter_and_finalize(
    all_results: list[dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    tg_target: float,
    out_dir: Path,
    max_abs_error: float = 25.0,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    banner("STEP 3/3  Physical Filtering and Final Ranking")
    print("Goal: validate generated structures and rank PI-plausible candidates.")
    print("Method: validator.py (L1 RDKit / L2 imide / L3 MW, heavy atoms, unstable groups).")
    print("input:   Step 1 generated SMILES and Step 2 DECIMER predictions")
    print(
        f"output:  PASS structures with |Step 2 predicted Tg - target| <= "
        f"{max_abs_error:.0f} deg C when available"
    )
    print()
    print("Filtering criteria:")
    print("  [L1] RDKit parse sanitize")
    print("  [L2] imide motif (-C(=O)-N-C(=O)-)")
    print("  [L3] molecular weight 150-2000; heavy atoms 10-150; no unstable groups")
    print(
        f"  [Tg] among physically passed rows, keep Top {top_k} by Tg error "
        f"within {max_abs_error:.0f} deg C when possible"
    )
    print()

    candidates: list[dict[str, Any]] = [r for r in all_results if r.get("smiles")]
    if not candidates:
        print("[Step3] No structures to validate; Step 1 did not produce valid SMILES.")
        return []

    _, validator_mod, _ = _load_inverse_stack()
    try:
        validate_smiles = validator_mod.validate_smiles
        print_report = validator_mod.print_report

        print(f"[Step3] Validating {len(candidates)} candidates ...")
        print()
        vr_list = []
        for row in candidates:
            mol_id = row.get("id", "?")
            smi = row["smiles"]
            print(f"-- validation {mol_id} --")
            vr = validate_smiles(smi, require_imide_ring=True, require_imide_carbonyl=True)
            print(vr.summary())
            print()
            vr_list.append((row, vr))

        print_report([vr for _, vr in vr_list])
    finally:
        _unload_inverse_stack()

    # Tg score: use Step 2 image prediction when available, otherwise use SMILES directly.
    from app.inference import get_predictor

    predictor = None

    def ensure_predictor():
        nonlocal predictor
        if predictor is None:
            print("[Step3] load TransPolymer imageprediction ...")
            predictor = get_predictor()
        return predictor

    scored_rows: list[dict[str, Any]] = []
    for row, vr in vr_list:
        if not vr.passed:
            continue
        mol_id = row.get("id", "?")
        true_smi = vr.smiles
        pred_rec = preds.get(mol_id, {})
        pred_tg = pred_rec.get("pred_tg_c")
        tg_source = pred_rec.get("tg_source") or "structure image DECIMER"

        if pred_tg is None:
            try:
                pred_tg = round(float(ensure_predictor().predict_tg(true_smi)), 2)
                tg_source = "SMILES"
                print(f"  [{mol_id}] Step 2 Tg missing; SMILES -> Tg={pred_tg:.2f} deg C")
            except Exception as exc:
                print(f"  [{mol_id}] Prediction failed: {exc}")
                continue

        err = round(float(pred_tg) - tg_target, 2)
        scored_rows.append(
            {
                "id": mol_id,
                "smiles": true_smi,
                "tg_target_c": tg_target,
                "pred_tg_c": float(pred_tg),
                "tg_error_c": err,
                "tg_source": tg_source,
                "mol_weight": vr.mol_weight,
                "num_heavy_atoms": vr.num_heavy_atoms,
                "imide_ring_count": vr.imide_ring_count,
                "num_rings": vr.num_rings,
                "parsed_smiles": pred_rec.get("parsed_smiles"),
                "image_path": pred_rec.get("image_path", ""),
            }
        )

    scored_rows.sort(key=lambda x: abs(x["tg_error_c"]))
    in_band = [r for r in scored_rows if abs(r["tg_error_c"]) <= max_abs_error]
    if in_band:
        final_rows = in_band[:top_k]
        band_note = f"|error|<={max_abs_error:.0f} deg C"
    else:
        final_rows = scored_rows[: min(top_k, len(scored_rows))]
        band_note = f"No rows within {max_abs_error:.0f} deg C; showing closest rows"
        print(f"\n[Step3] {band_note}. Consider increasing --n_samples.")

    n_phys = len(scored_rows)
    subbanner(
        f"Final candidates: physically passed {n_phys}/{len(candidates)}, "
        f"in error band {len(in_band)}, shown {len(final_rows)}; {band_note}"
    )
    if not final_rows:
        print("No structures passed filtering. Try increasing --n_samples or --max_abs_error.")
    else:
        print(
            f"{'rank':<4} {'ID':<10} {'pred_Tg':>8} {'error':>8} {'source':<20} SMILES"
        )
        print("-" * BANNER_W)
        for i, r in enumerate(final_rows, 1):
            print(
                f"{i:<4} {r['id']:<10} {r['pred_tg_c']:>8.2f} "
                f"{r['tg_error_c']:>+8.2f} {str(r['tg_source']):<20} "
                f"{short_smiles(r['smiles'], 40)}"
            )
            print(
                f"     PASS details: MW={r['mol_weight']}  "
                f"heavy atoms={r['num_heavy_atoms']}  "
                f"imide rings={r['imide_ring_count']}  "
                f"predicted Tg={r['pred_tg_c']:.2f} deg C (source: {r['tg_source']})  "
                f"|error|={abs(r['tg_error_c']):.2f}"
            )

    final_csv = out_dir / "final.csv"
    fieldnames = [
        "rank",
        "id",
        "smiles",
        "tg_target_c",
        "pred_tg_c",
        "tg_error_c",
        "tg_source",
        "mol_weight",
        "num_heavy_atoms",
        "imide_ring_count",
        "num_rings",
        "parsed_smiles",
        "image_path",
    ]
    with final_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(final_rows, 1):
            writer.writerow({"rank": i, **r})

    step2_dump = [
        {
            "id": v["id"],
            "true_smiles": v.get("true_smiles"),
            "parsed_smiles": v.get("parsed_smiles"),
            "pred_tg_c": v.get("pred_tg_c"),
            "tg_source": v.get("tg_source"),
            "error": v.get("error"),
            "image_path": v.get("image_path"),
        }
        for v in preds.values()
    ]
    summary = {
        "tg_target_c": tg_target,
        "max_abs_error_c": max_abs_error,
        "top_k": top_k,
        "n_generated": len(all_results),
        "n_rdkit_valid": sum(1 for r in all_results if r.get("smiles")),
        "n_image_pred_ok": sum(1 for v in preds.values() if v.get("pred_tg_c") is not None),
        "n_physically_passed": n_phys,
        "n_in_error_band": len(in_band),
        "final_csv": str(final_csv),
        "structures_dir": str(out_dir / "structures"),
        "step2_predictions": step2_dump,
        "final": final_rows,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"[Step3] Final CSV: {final_csv}")
    print(f"[Step3] Summary JSON: {summary_path}")
    return final_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PI three-step demo: generate structures -> image-based Tg prediction -> physical filtering"
    )
    p.add_argument("--tg_target", type=float, default=300.0, help="Target Tg in deg C")
    p.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of candidates to generate",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory; default is results/demo_YYYYMMDD_HHMMSS",
    )
    p.add_argument("--ckpt", type=str, default="", help="Inverse generator checkpoint path")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--max_abs_error",
        type=float,
        default=25.0,
        help="Maximum |predicted Tg - target Tg| in deg C",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of final rows to report after sorting by absolute error",
    )
    p.add_argument(
        "--pause",
        action="store_true",
        help="Pause for Enter between steps",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tg_target = round(float(args.tg_target), 1)
    ckpt = Path(args.ckpt) if args.ckpt else Path(
        os.environ.get("PI_INVERSE_CKPT", str(DEFAULT_INVERSE_CKPT))
    )
    if not ckpt.is_file():
        print(f"Inverse checkpoint does not exist: {ckpt}", file=sys.stderr)
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / f"demo_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results, structures_dir = step1_generate(
        tg_target=tg_target,
        n_samples=args.n_samples,
        ckpt=ckpt,
        out_dir=out_dir,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    maybe_pause(args.pause, "Step1 complete")

    preds = step2_image_predict(all_results, structures_dir)
    maybe_pause(args.pause, "Step2 complete")

    final_rows = step3_filter_and_finalize(
        all_results=all_results,
        preds=preds,
        tg_target=tg_target,
        out_dir=out_dir,
        max_abs_error=float(args.max_abs_error),
        top_k=int(args.top_k),
    )

    banner("")
    print(f"Rows passing physical filtering: {len(final_rows)}")
    print(f"Output directory: {out_dir}")
    print("  - structures/     structure image PNG")
    print("  - generated_all.csv")
    print("  - final.csv       filtered plausible structures + Tg")
    print("  - summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
