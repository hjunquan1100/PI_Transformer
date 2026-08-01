

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
from sklearn.metrics import r2_score

from generate  import load_model, generate_batch
from forward_model import resolve_forward_model_path
from paths import DEFAULT_FORWARD_MODEL, DEFAULT_GENERATOR_CKPT, DEFAULT_PROCESSED_DIR


def compute_validity(results: list) -> float:
    """Return the fraction of generated rows with RDKit-valid SMILES."""
    valid = sum(1 for r in results if r["valid"])
    return valid / len(results) if results else 0.0


def compute_uniqueness(results: list) -> float:
    """Return unique valid SMILES divided by all valid SMILES."""
    valid_smiles = [r["smiles"] for r in results if r["valid"] and r["smiles"]]
    if not valid_smiles:
        return 0.0
    unique = set(valid_smiles)
    return len(unique) / len(valid_smiles)


def compute_novelty(results: list, train_smiles_set: set) -> float:
    """Return valid SMILES not present in the training set divided by valid SMILES."""
    valid_smiles = [r["smiles"] for r in results if r["valid"] and r["smiles"]]
    if not valid_smiles:
        return 0.0
    novel = sum(1 for s in valid_smiles if s not in train_smiles_set)
    return novel / len(valid_smiles)


def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """Convert SMILES to a Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def compute_internal_diversity(results: list, sample_n: int = 100) -> float:
    """
    Estimate internal diversity as 1 - mean sampled Tanimoto similarity.
    """
    valid_smiles = list({r["smiles"] for r in results if r["valid"] and r["smiles"]})
    if len(valid_smiles) < 2:
        return 0.0

    fps = [smiles_to_fp(s) for s in valid_smiles]
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return 0.0

    rng = np.random.default_rng(0)
    n   = min(len(fps), sample_n)
    idx = rng.choice(len(fps), size=(n, 2), replace=True)
    sims = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i, j in idx if i != j
    ]
    mean_sim = float(np.mean(sims)) if sims else 0.0
    return 1.0 - mean_sim


def compute_tg_hit_rate(
    results: list,
    tg_target: float,
    fwd_model,
    tolerance: float = 30.0,
) -> dict:
    """
    Score generated molecules with the forward Tg model.

    Hit rate is the fraction with |predicted Tg - target Tg| < tolerance.
    """
    valid_smiles = list({r["smiles"] for r in results if r["valid"] and r["smiles"]})
    if not valid_smiles or fwd_model is None:
        return {"hit_rate": None, "mae": None, "valid_count": len(valid_smiles)}

    fps  = [smiles_to_fp(s) for s in valid_smiles]
    mask = [f is not None for f in fps]
    fps_clean  = [f for f, m in zip(fps, mask) if m]
    smi_clean  = [s for s, m in zip(valid_smiles, mask) if m]

    if not fps_clean:
        return {"hit_rate": None, "mae": None, "valid_count": 0}

    X       = np.array([list(fp) for fp in fps_clean])
    tg_preds = fwd_model.predict(X)

    errors   = np.abs(tg_preds - tg_target)
    hit_rate = float(np.mean(errors < tolerance))
    mae      = float(np.mean(errors))

    return {
        "hit_rate":    round(hit_rate, 4),
        "mae":         round(mae, 2),
        "valid_count": len(fps_clean),
        "tg_preds":    tg_preds.tolist(),
    }


def compute_overall_tg_r2(true_tg: list, pred_tg: list):
    """Compute overall Tg R2 across all targets when at least two targets exist."""
    if len(true_tg) < 2:
        return None
    if len(set(true_tg)) < 2:
        return None
    return float(r2_score(true_tg, pred_tg))


def evaluate_model(
    model,
    vocab:          dict,
    inv_vocab:      dict,
    stats:          dict,
    tg_targets:     list,
    train_smiles:   list    = None,
    fwd_model               = None,
    n_samples:      int     = 100,
    temperature:    float   = 1.0,
    top_p:          float   = 0.9,
    tolerance:      float   = 30.0,
    device:         str     = "cpu",
    out_path:       str     = None,
):
    """Evaluate generation quality over a list of target Tg values."""
    train_smiles_set = set(train_smiles) if train_smiles else set()
    report = []
    overall_true_tg = []
    overall_pred_tg = []

    for tg in tg_targets:
        print(f"\n{'-'*50}")
        print(f"Evaluating target Tg = {tg} deg C")

        output = generate_batch(
            model       = model,
            tg_target   = tg,
            vocab       = vocab,
            inv_vocab   = inv_vocab,
            stats       = stats,
            n_samples   = n_samples,
            temperature = temperature,
            top_p       = top_p,
            device      = device,
            validate    = True,
            strict      = True,
            verbose     = True,
        )

        results         = output["all_results"]
        passed_results  = output["passed_results"]

        validity   = compute_validity(results)
        pi_pass    = round(len(passed_results) / max(output["valid_count"], 1), 4)
        uniqueness = compute_uniqueness(passed_results)
        novelty    = compute_novelty(passed_results, train_smiles_set)
        diversity  = compute_internal_diversity(passed_results)
        tg_metrics = compute_tg_hit_rate(passed_results, tg, fwd_model, tolerance)

        tg_preds = tg_metrics.get("tg_preds") or []
        overall_true_tg.extend([tg] * len(tg_preds))
        overall_pred_tg.extend(tg_preds)

        row = {
            "tg_target":    tg,
            "n_samples":    n_samples,
            "rdkit_valid":  round(validity,  4),
            "pi_pass_rate": pi_pass,
            "uniqueness":  round(uniqueness, 4),
            "novelty":     round(novelty,    4),
            "diversity":   round(diversity,  4),
            "tg_hit_rate": tg_metrics.get("hit_rate"),
            "tg_mae":      tg_metrics.get("mae"),
        }
        report.append(row)

        print(f"\n  Tg={tg} deg C evaluation:")
        print(f"     RDKit valid:      {validity*100:.1f}%")
        print(f"     PI plausible:     {pi_pass*100:.1f}%")

        print(f"     uniqueness:       {uniqueness*100:.1f}%  (target >70%)")
        print(f"     novelty:          {novelty*100:.1f}%  (target >50%)")
        print(f"     diversity:        {diversity:.3f}  (higher is better)")
        if tg_metrics.get("hit_rate") is not None:
            print(f"     Tg hit rate:      {tg_metrics['hit_rate']*100:.1f}%  (+/-{tolerance} deg C)")
            print(f"     Tg MAE:           {tg_metrics['mae']:.1f} deg C")

    df = pd.DataFrame(report)
    overall_tg_r2 = compute_overall_tg_r2(overall_true_tg, overall_pred_tg)
    print(f"\n{'='*60}")
    print("Evaluation summary:")
    print(df.to_string(index=False))
    if overall_tg_r2 is not None:
        print(f"\nTg R2: {overall_tg_r2:.4f}")
    else:
        print("\nTg R2: unavailable; at least two targets with valid predictions are required.")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nEvaluation CSV saved: {out_path}")
        summary_path = out_path.replace(".csv", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "overall_tg_r2": None if overall_tg_r2 is None else round(overall_tg_r2, 4),
                "overall_valid_predictions": len(overall_pred_tg),
            }, f, indent=2, ensure_ascii=False)
        print(f"Evaluation summary saved: {summary_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the PI inverse-design model")
    parser.add_argument("--ckpt",       default=str(DEFAULT_GENERATOR_CKPT))
    parser.add_argument("--fwd_model",  default=str(DEFAULT_FORWARD_MODEL),
                        help="Forward Tg prediction model path (optional)")
    parser.add_argument("--train_data", default=str(DEFAULT_PROCESSED_DIR / "val.pkl"),
                        help="Processed training/validation pkl used for novelty scoring")
    parser.add_argument("--tg_targets", nargs="+", type=float,
                        default=[200, 250, 300, 350, 400])
    parser.add_argument("--n_samples",  type=int,   default=100)
    parser.add_argument("--temperature",type=float, default=1.0)
    parser.add_argument("--top_p",      type=float, default=0.9)
    parser.add_argument("--tolerance",  type=float, default=30.0)
    parser.add_argument("--out",        default="results/evaluation_report.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab, inv_vocab, stats = load_model(args.ckpt, device=device)

    fwd_model = None
    try:
        fwd_model_path = resolve_forward_model_path(args.fwd_model)
    except FileNotFoundError:
        fwd_model_path = None

    if fwd_model_path:
        with open(fwd_model_path, "rb") as f:
            fwd_model = pickle.load(f)
        print(f"Forward prediction model loaded: {fwd_model_path}")
    else:
        print("Forward prediction model not found; Tg hit-rate evaluation will be skipped.")

    train_smiles = []
    if os.path.exists(args.train_data):
        import pickle as pk
        with open(args.train_data, "rb") as f:
            data = pk.load(f)
        try:
            import selfies as sf
            from rdkit import Chem
            for sf_str, _ in data:
                smi = sf.decoder(sf_str)
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    train_smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            pass
        print(f"Training SMILES loaded for novelty scoring: {len(train_smiles)}")

    evaluate_model(
        model        = model,
        vocab        = vocab,
        inv_vocab    = inv_vocab,
        stats        = stats,
        tg_targets   = args.tg_targets,
        train_smiles = train_smiles,
        fwd_model    = fwd_model,
        n_samples    = args.n_samples,
        temperature  = args.temperature,
        top_p        = args.top_p,
        tolerance    = args.tolerance,
        device       = device,
        out_path     = args.out,
    )
