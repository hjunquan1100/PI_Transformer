# -*- coding: utf-8 -*-

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_holdout", type=int, default=80, help="Number of holdout rows")
    ap.add_argument("--seed", type=int, default=42, help="random seed, for reproducibility")
    args = ap.parse_args()

    base_path = DATA / "PI_Tg.csv"
    desc_path = DATA / "PI_Tg_with_desc.csv"

    df_base = pd.read_csv(base_path)
    df_desc = pd.read_csv(desc_path)
    if len(df_base) != len(df_desc):
        raise SystemExit(
            "row count mismatch: PI_Tg.csv (%d) vs PI_Tg_with_desc.csv (%d)"
            % (len(df_base), len(df_desc))
        )

    n = len(df_base)
    if args.n_holdout >= n or args.n_holdout < 1:
        raise SystemExit("n_holdout must be in [1, %d) " % n)

    idx = df_base.index.values
    pool_idx, hold_idx = train_test_split(
        idx,
        test_size=args.n_holdout,
        random_state=args.seed,
        shuffle=True,
    )

    hold_idx = sorted(hold_idx.tolist())
    pool_idx = sorted(pool_idx.tolist())

    df_base.iloc[pool_idx].to_csv(DATA / "PI_Tg_train_pool.csv", index=False)
    df_base.iloc[hold_idx].to_csv(DATA / "PI_Tg_holdout80.csv", index=False)
    df_desc.iloc[pool_idx].to_csv(DATA / "PI_Tg_with_desc_train_pool.csv", index=False)
    df_desc.iloc[hold_idx].to_csv(DATA / "PI_Tg_with_desc_holdout80.csv", index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_total": int(n),
        "n_train_pool": int(len(pool_idx)),
        "n_holdout": int(len(hold_idx)),
        "random_state": args.seed,
        "holdout_indices_0based": hold_idx,
        "outputs": {
            "train_pool_smiles_only": "data/PI_Tg_train_pool.csv",
            "holdout_smiles_only": "data/PI_Tg_holdout80.csv",
            "train_pool_with_desc": "data/PI_Tg_with_desc_train_pool.csv",
            "holdout_with_desc": "data/PI_Tg_with_desc_holdout80.csv",
        },
    }
    with open(DATA / "PI_Tg_holdout_split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("n_total=%d train_pool=%d holdout=%d seed=%d" % (n, len(pool_idx), len(hold_idx), args.seed))
    print("Wrote data/PI_Tg_train_pool.csv, data/PI_Tg_holdout80.csv,")
    print("     data/PI_Tg_with_desc_train_pool.csv, data/PI_Tg_with_desc_holdout80.csv,")
    print("     data/PI_Tg_holdout_split_meta.json")


if __name__ == "__main__":
    main()
