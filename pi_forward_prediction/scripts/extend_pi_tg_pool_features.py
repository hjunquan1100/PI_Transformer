#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED

ROOT = Path(__file__).resolve().parents[1]

SCALAR_FUNCS = [
    ("MolLogP", lambda m: float(Descriptors.MolLogP(m))),
    ("NumRadicalElectrons", lambda m: float(Descriptors.NumRadicalElectrons(m))),
    ("NumHeteroatoms", lambda m: float(Lipinski.NumHeteroatoms(m))),
    ("NumAliphaticRings", lambda m: float(Lipinski.NumAliphaticRings(m))),
    ("NumSaturatedRings", lambda m: float(Lipinski.NumSaturatedRings(m))),
    ("qed", lambda m: float(QED.qed(m))),
]


def featurize(smiles: str, morgan_bits: int):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    out = {}
    if mol is None:
        for name, _ in SCALAR_FUNCS:
            out[name] = 0.0
        for i in range(morgan_bits):
            out["mfp_%d" % i] = 0.0
        return out
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        for name, _ in SCALAR_FUNCS:
            out[name] = 0.0
        for i in range(morgan_bits):
            out["mfp_%d" % i] = 0.0
        return out
    for name, fn in SCALAR_FUNCS:
        try:
            out[name] = fn(mol)
        except Exception:
            out[name] = 0.0
    if morgan_bits > 0:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits)
        for i in range(morgan_bits):
            out["mfp_%d" % i] = float(fp[i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default="data/PI_Tg_with_desc_train_pool.csv")
    ap.add_argument("--out", type=str, default="data/PI_Tg_with_desc_train_pool_p1_ext.csv")
    ap.add_argument("--morgan-bits", type=int, default=64)
    ap.add_argument("--manifest", type=str, default="experiments/results/p1_feature_manifest.json")
    args = ap.parse_args()

    inp = Path(args.in_path)
    if not inp.is_absolute():
        inp = ROOT / inp
    df = pd.read_csv(inp)
    if "smiles" not in df.columns:
        raise SystemExit("need smiles column")

    extra_cols = [n for n, _ in SCALAR_FUNCS] + (["mfp_%d" % i for i in range(args.morgan_bits)] if args.morgan_bits > 0 else [])
    rows = []
    for _, row in df.iterrows():
        d = featurize(row["smiles"], int(args.morgan_bits))
        rows.append(d)
    ext = pd.DataFrame(rows)
    out_df = pd.concat([df.reset_index(drop=True), ext], axis=1)

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outp, index=False)

    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    man_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(inp.relative_to(ROOT)) if str(inp).startswith(str(ROOT)) else str(inp),
        "output_csv": str(outp.relative_to(ROOT)) if str(outp).startswith(str(ROOT)) else str(outp),
        "morgan_bits": int(args.morgan_bits),
        "added_scalar_columns": [n for n, _ in SCALAR_FUNCS],
        "added_morgan_columns": ["mfp_%d" % i for i in range(int(args.morgan_bits))] if args.morgan_bits > 0 else [],
        "descriptor_cols_for_yaml": [c for c in out_df.columns if c not in ("smiles", "value")],
        "n_rows": int(len(out_df)),
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Wrote", outp, "cols+", len(extra_cols), "manifest", man_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
