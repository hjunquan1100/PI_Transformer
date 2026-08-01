

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from paths import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DATA

try:
    import selfies as sf
except ImportError:
    raise ImportError("Please install selfies: pip install selfies")

def canonicalize(smiles: str):
    """Return canonical SMILES, or None when RDKit parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_selfies(smiles: str):
    """Convert SMILES to SELFIES, or None on failure."""
    try:
        canon = canonicalize(smiles)
        if canon is None:
            return None
        return sf.encoder(canon)
    except Exception:
        return None


def selfies_to_smiles(selfies_str: str):
    """Convert SELFIES to SMILES."""
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return None


def augment_molecule(smiles: str, n: int = 8):
    """
    Generate up to n randomized SELFIES strings from one SMILES string.

    This uses RDKit SMILES enumeration for data augmentation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    results = set()
    max_tries = n * 5
    for _ in range(max_tries):
        try:
            rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            sf_str = sf.encoder(rand_smi)
            if sf_str:
                results.add(sf_str)
        except Exception:
            continue
        if len(results) >= n:
            break
    return list(results)


def run_preprocess(
    data_path: str  = str(DEFAULT_RAW_DATA),
    out_dir: str    = str(DEFAULT_PROCESSED_DIR),
    augment_n: int  = 8,       # augmented samples per molecule
    n_bins: int     = 20,      # Tg binning
    val_ratio: float= 0.1,
    seed: int       = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"[1/6] Loading raw data: {data_path}")
    df = pd.read_excel(data_path)
    print(f"      Raw rows: {len(df)}")

    smiles_col = "SMILES-Repeating unit"
    tg_col     = "Tg( deg C)"
    df = df[[smiles_col, tg_col]].dropna()
    print(f"      Rows with SMILES and Tg: {len(df)}")

    print("[2/6] Sanitizing SMILES and removing duplicates")
    df["canon_smiles"] = df[smiles_col].apply(canonicalize)
    before = len(df)
    df = df.dropna(subset=["canon_smiles"])
    df = df.drop_duplicates(subset=["canon_smiles"])
    print(f"      Valid unique molecules: {len(df)}  (removed {before - len(df)})")

    print("[3/6] Converting SMILES to SELFIES")
    df["selfies"] = df["canon_smiles"].apply(smiles_to_selfies)
    df = df.dropna(subset=["selfies"])
    print(f"      Successful conversions: {len(df)}")

    print("[4/6] Creating train/validation split")
    tg_values = df[tg_col].values
    # Stratify approximately by Tg range.
    bin_labels = pd.cut(tg_values, bins=10, labels=False)
    val_indices, train_indices = [], []
    for b in range(10):
        idx = np.where(bin_labels == b)[0]
        if len(idx) == 0:
            continue
        n_val = max(1, int(len(idx) * val_ratio))
        chosen = rng.choice(idx, size=n_val, replace=False)
        val_indices.extend(chosen)
        train_indices.extend([i for i in idx if i not in set(chosen)])

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df   = df.iloc[val_indices].reset_index(drop=True)
    print(f"      Train rows: {len(train_df)}  validation rows: {len(val_df)}")

    print("[5/6] Computing Tg statistics")
    tg_train   = train_df[tg_col].values.astype(float)
    tg_mean    = float(np.mean(tg_train))
    tg_std     = float(np.std(tg_train))
    tg_min     = float(np.min(tg_train))
    tg_max_val = float(np.max(tg_train))
    tg_bins    = np.linspace(tg_min - 1, tg_max_val + 1, n_bins + 1).tolist()

    stats = {
        "tg_mean": tg_mean,
        "tg_std":  tg_std,
        "tg_min":  tg_min,
        "tg_max":  tg_max_val,
        "tg_bins": tg_bins,
        "n_bins":  n_bins,
    }
    print(f"      Tg mean={tg_mean:.1f}  std={tg_std:.1f}  range=[{tg_min:.0f}, {tg_max_val:.0f}]")

    print(f"[6/6] Building augmented training data (x{augment_n})")
    augmented_data = []
    for _, row in train_df.iterrows():
        smi = row["canon_smiles"]
        tg  = float(row[tg_col])
        augmented_data.append((row["selfies"], tg))
        extra = augment_molecule(smi, n=augment_n - 1)
        for sf_str in extra:
            augmented_data.append((sf_str, tg))

    print(f"      Augmented training samples: {len(augmented_data)}")

    val_data = [
        (row["selfies"], float(row[tg_col]))
        for _, row in val_df.iterrows()
    ]

    print("      Building SELFIES vocabulary")
    token_counter = Counter()
    for sf_str, _ in augmented_data:
        token_counter.update(sf.split_selfies(sf_str))

    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    all_tokens     = special_tokens + sorted(token_counter.keys())
    vocab          = {tok: i for i, tok in enumerate(all_tokens)}
    inv_vocab      = {i: tok for tok, i in vocab.items()}
    print(f"      Vocabulary size: {len(vocab)}")

    with open(os.path.join(out_dir, "train_augmented.pkl"), "wb") as f:
        pickle.dump(augmented_data, f)
    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "inv_vocab": {str(k): v for k, v in inv_vocab.items()}}, f, indent=2)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nPreprocessing complete. Files saved to:", out_dir)
    print(f"  train_augmented.pkl  {len(augmented_data)} items")
    print(f"  val.pkl              {len(val_data)} items")
    print(f"  vocab.json           {len(vocab)} tokens")
    print(f"  stats.json           tg_mean={tg_mean:.2f}")
    return vocab, inv_vocab, stats, augmented_data, val_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--out_dir",   default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--augment_n", type=int,   default=8)
    parser.add_argument("--n_bins",    type=int,   default=20)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    run_preprocess(
        data_path  = args.data,
        out_dir    = args.out_dir,
        augment_n  = args.augment_n,
        n_bins     = args.n_bins,
        val_ratio  = args.val_ratio,
        seed       = args.seed,
    )
