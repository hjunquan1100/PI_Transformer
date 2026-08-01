# -*- coding: utf-8 -*-

import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

DESC_COLS = [
    "NumRotatableBonds",
    "HeavyAtomCount",
    "MolWt",
    "FractionCSP3",
    "NumAromaticRings",
    "RingCount",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
]


def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return {
        "NumRotatableBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        "MolWt": float(Descriptors.MolWt(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "NumAromaticRings": float(Lipinski.NumAromaticRings(mol)),
        "RingCount": float(Lipinski.RingCount(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
    }


def main():
    in_path = "data/PI_Tg.csv"
    out_path = "data/PI_Tg_with_desc.csv"
    df = pd.read_csv(in_path)
    assert "smiles" in df.columns and "value" in df.columns, df.columns.tolist()

    rows = []
    failed = []
    for i, row in df.iterrows():
        smi = row["smiles"]
        d = compute_descriptors(smi)
        if d is None:
            failed.append((i, str(smi)[:120]))
            continue
        rows.append({"smiles": smi, "value": row["value"], **d})

    if failed:
        print(
            f"[prep_PI_Tg_descriptors] dropped {len(failed)} / {len(df)} rows (RDKit parse/sanitize failed)",
            file=sys.stderr,
        )
        for i, (idx, s) in enumerate(failed[:5]):
            print(f"  example {i + 1} idx={idx}: {s!r}", file=sys.stderr)

    out = pd.DataFrame(rows)
    out = out[["smiles", "value"] + DESC_COLS]
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}, n = {len(out)} (from {len(df)} input)")


if __name__ == "__main__":
    main()
