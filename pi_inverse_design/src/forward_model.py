

import os
import argparse
import pickle
import json
import time
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from paths import DEFAULT_FORWARD_MODEL, DEFAULT_FORWARD_MODEL_OUT, DEFAULT_RAW_DATA


def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """Convert a SMILES string to a Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def build_features(smiles_list: list, radius: int = 2, nbits: int = 2048):
    """Build fingerprint features and return (X, valid_mask)."""
    fps, mask = [], []
    for smi in smiles_list:
        fp = smiles_to_fp(smi, radius, nbits)
        if fp is not None:
            fps.append(fp)
            mask.append(True)
        else:
            fps.append(np.zeros(nbits, dtype=np.int8))
            mask.append(False)
    return np.array(fps), np.array(mask)


def resolve_forward_model_path(model_path: str) -> str:
    """Resolve a forward-model path, including timestamped training outputs."""
    if os.path.exists(model_path):
        return model_path

    model_dir = os.path.dirname(model_path) or "."
    model_name = os.path.basename(model_path)
    if model_name != "forward_model.pkl" or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Forward model not found: {model_path}")

    candidates = []
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name.startswith("forward_model_best_r2_") and name.endswith(".pkl"):
                candidates.append(os.path.join(root, name))

    if not candidates:
        raise FileNotFoundError(f"Forward model not found: {model_path}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def train_forward_model(
    data_path:  str = str(DEFAULT_RAW_DATA),
    out_path:   str = str(DEFAULT_FORWARD_MODEL_OUT),
    radius:     int = 2,
    nbits:      int = 2048,
    n_estimators: int = 500,
    max_depth:    int = 4,
    seed:         int = 42,
):
    base_out_dir = os.path.dirname(out_path)
    os.makedirs(base_out_dir, exist_ok=True)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join(base_out_dir, f"forward_{run_timestamp}")
    os.makedirs(run_out_dir, exist_ok=True)
    out_path = os.path.join(run_out_dir, os.path.basename(out_path))
    print(f"Forward model output directory: {run_out_dir}")

    print(f"[1/4] Loading data: {data_path}")
    df = pd.read_excel(data_path)
    smiles_col = "SMILES-Repeating unit"
    tg_col     = "Tg( deg C)"
    df = df[[smiles_col, tg_col]].dropna()

    canon_list, tg_list = [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        canon_list.append(Chem.MolToSmiles(mol))
        tg_list.append(float(row[tg_col]))

    print(f"      Valid molecules: {len(canon_list)}")

    print(f"[2/4] Building Morgan fingerprints (radius={radius}, nbits={nbits})")
    X, mask = build_features(canon_list, radius, nbits)
    y = np.array(tg_list)
    X, y = X[mask], y[mask]
    print(f"      Feature matrix: {X.shape}")

    print(f"[3/4] Training GradientBoostingRegressor (n_estimators={n_estimators})")
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("gbr",    GradientBoostingRegressor(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            learning_rate= 0.05,
            subsample    = 0.8,
            random_state = seed,
        ))
    ])

    kf     = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_mae = []
    cv_r2 = []
    cv_start = time.time()
    best_fold = None
    best_r2 = float("-inf")
    best_mae = None
    best_model = None
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        fold_start = time.time()
        fold_model = clone(model)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(
            f"      Fold {fold}/5: "
            f"train={len(train_idx)} val={len(val_idx)} ..."
        )
        fold_model.fit(X_train, y_train)
        val_pred = fold_model.predict(X_val)

        fold_mae = mean_absolute_error(y_val, val_pred)
        fold_r2 = r2_score(y_val, val_pred)
        cv_mae.append(fold_mae)
        cv_r2.append(fold_r2)
        if fold_r2 > best_r2:
            best_fold = fold
            best_r2 = fold_r2
            best_mae = fold_mae
            best_model = fold_model

        print(
            f"        complete Fold {fold}/5 | "
            f"MAE={fold_mae:.2f} deg C | "
            f"R2={fold_r2:.4f} | "
            f"{time.time() - fold_start:.1f}s"
        )

    cv_mae = np.array(cv_mae)
    cv_r2  = np.array(cv_r2)
    print(f"      validation: {time.time() - cv_start:.1f}s")

    print(f"\n  5-fold CV MAE: {cv_mae.mean():.2f} +/- {cv_mae.std():.2f} deg C")
    print(f"  5-fold CV R2:  {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"  Best fold:     {best_fold} | MAE={best_mae:.2f} deg C | R2={best_r2:.4f}")

    if cv_mae.mean() > 30:
        print("  Warning: MAE > 30 deg C. Consider tuning features or n_estimators.")

    full_train_start = time.time()
    print("      Training final model on all data")
    model.fit(X, y)
    train_pred = model.predict(X)
    train_mae  = mean_absolute_error(y, train_pred)
    train_r2   = r2_score(y, train_pred)
    print(f"      training: {time.time() - full_train_start:.1f}s")
    print(f"\n  Training MAE: {train_mae:.2f} deg C")
    print(f"  Training R2:  {train_r2:.4f}")

    out_dir = os.path.dirname(out_path)
    out_name = os.path.splitext(os.path.basename(out_path))[0]
    out_ext = os.path.splitext(out_path)[1] or ".pkl"
    best_model_path = os.path.join(out_dir, f"{out_name}_best_r2_{best_r2:.4f}{out_ext}")

    print(f"\n[4/4] Saving best-R2 model: {best_model_path}")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"      Stable model path: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(best_model, f)

    metrics = {
        "cv_mae_mean": round(float(cv_mae.mean()), 3),
        "cv_mae_std":  round(float(cv_mae.std()),  3),
        "cv_r2_mean":  round(float(cv_r2.mean()),  4),
        "best_fold":   int(best_fold),
        "best_mae":    round(float(best_mae),       3),
        "best_r2":     round(float(best_r2),        4),
        "best_model_path": best_model_path,
        "train_mae":   round(float(train_mae),      3),
        "train_r2":    round(float(train_r2),       4),
        "n_samples":   len(y),
        "radius":      radius,
        "nbits":       nbits,
    }
    metrics_path = out_path.replace(".pkl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved: {metrics_path}")
    return model, metrics


def predict_tg(smiles_list: list, model_path: str = str(DEFAULT_FORWARD_MODEL)):
    """Predict Tg values for a list of SMILES strings."""
    model_path = resolve_forward_model_path(model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X, mask = build_features(smiles_list)
    preds   = np.full(len(smiles_list), np.nan)
    if mask.any():
        preds[mask] = model.predict(X[mask])
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Tg prediction model")
    parser.add_argument("--data",        default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--out",         default=str(DEFAULT_FORWARD_MODEL_OUT))
    parser.add_argument("--radius",      type=int, default=2)
    parser.add_argument("--nbits",       type=int, default=2048)
    parser.add_argument("--n_estimators",type=int, default=500)
    parser.add_argument("--predict",     type=str, default=None,
                        help="Input SMILES for single-molecule prediction")
    args = parser.parse_args()

    if args.predict:
        preds = predict_tg([args.predict], model_path=args.out)
        print(f"Predicted Tg: {preds[0]:.1f} deg C")
    else:
        train_forward_model(
            data_path    = args.data,
            out_path     = args.out,
            radius       = args.radius,
            nbits        = args.nbits,
            n_estimators = args.n_estimators,
        )
