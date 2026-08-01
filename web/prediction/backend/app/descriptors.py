"""Compute RDKit descriptors for a single polymer SMILES (PI Tg model)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import PI_FORWARD_ROOT

if str(PI_FORWARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PI_FORWARD_ROOT))

from prep_PI_Tg_descriptors import compute_descriptors as compute_base_descriptors  # noqa: E402

# Reuse extend script featurize for extra scalars + Morgan bits
_extend_script = PI_FORWARD_ROOT / "scripts" / "extend_pi_tg_pool_features.py"
if _extend_script.is_file():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("extend_pi_tg_pool_features", _extend_script)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    featurize_extended = _mod.featurize
else:
    raise RuntimeError(f"Missing extend script: {_extend_script}")


class SmilesDescriptorError(ValueError):
    """SMILES cannot be parsed or sanitized for descriptor computation."""


# PNG text key used to restore the same SMILES after saving and re-uploading a structure image.
PNG_SMILES_KEY = "pi_smiles"


def canonicalize_smiles(smiles: str) -> str:
    """Parse with RDKit and return canonical SMILES."""
    from rdkit import Chem

    s = str(smiles).strip()
    if not s:
        raise SmilesDescriptorError("SMILES cannot be empty.")
    mol = Chem.MolFromSmiles(s, sanitize=True)
    if mol is None:
        raise SmilesDescriptorError(
            "Cannot parse or sanitize this SMILES. Check the polyimide repeat-unit syntax; "
            "* connection points are allowed."
        )
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise SmilesDescriptorError(f"Structure sanitization failed: {exc}") from exc
    return Chem.MolToSmiles(mol, canonical=True)


def compute_all_descriptors(smiles: str, morgan_bits: int = 128) -> dict[str, float]:
    smiles = canonicalize_smiles(smiles)
    base = compute_base_descriptors(smiles)
    if base is None:
        raise SmilesDescriptorError(
            "Cannot parse or sanitize this SMILES. Check the polyimide repeat-unit syntax; "
            "* connection points are allowed."
        )
    extra = featurize_extended(smiles, morgan_bits)
    merged: dict[str, float] = {**base, **extra}
    return merged


def row_for_descriptor_cols(smiles: str, desc_cols: list[str], morgan_bits: int = 128) -> pd.Series:
    feats = compute_all_descriptors(smiles, morgan_bits=morgan_bits)
    missing = [c for c in desc_cols if c not in feats]
    if missing:
        raise SmilesDescriptorError(f"Descriptor columns are missing: {missing[:5]}")
    return pd.Series({c: float(feats[c]) for c in desc_cols})


def molecule_identity_key(smiles: str) -> str | None:
    """Build a stable identity key for same-compound matching."""
    from rdkit import Chem

    try:
        canon = canonicalize_smiles(smiles)
    except SmilesDescriptorError:
        return None
    # Cap polymer connection points before InChI comparison.
    capped = canon.replace("*", "[H]")
    mol = Chem.MolFromSmiles(capped, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles(canon.replace("*", ""), sanitize=True)
    if mol is None:
        return canon
    try:
        key = Chem.MolToInchiKey(mol)
        if key:
            return key
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def same_compound(smiles_a: str, smiles_b: str, fp_similarity_min: float = 0.95) -> bool:
    """
    Return whether two SMILES strings represent the same or highly similar compound.
    InChIKey/canonical equality is preferred; otherwise Morgan similarity is used.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    ka = molecule_identity_key(smiles_a)
    kb = molecule_identity_key(smiles_b)
    if ka and kb and ka == kb:
        return True

    def _fp(s: str):
        try:
            c = canonicalize_smiles(s).replace("*", "[H]")
        except SmilesDescriptorError:
            return None
        mol = Chem.MolFromSmiles(c)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    fa, fb = _fp(smiles_a), _fp(smiles_b)
    if fa is None or fb is None:
        return False
    sim = float(DataStructs.TanimotoSimilarity(fa, fb))
    return sim >= fp_similarity_min



def smiles_to_svg(smiles: str, width: int = 400, height: int = 300) -> str:
    from rdkit import Chem
    from rdkit.Chem import Draw

    smiles = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise SmilesDescriptorError("Cannot generate a 2D structure from SMILES.")
    drawer = Draw.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def smiles_to_png_bytes(smiles: str, width: int = 1200, height: int = 400) -> bytes:
    """Generate a PNG with embedded pi_smiles metadata."""
    from io import BytesIO

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    canon = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canon, sanitize=True)
    if mol is None:
        raise SmilesDescriptorError("Cannot generate a 2D structure from SMILES.")
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(width, height), kekulize=True)
    meta = PngInfo()
    meta.add_text(PNG_SMILES_KEY, canon)
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def smiles_from_png_bytes(image_bytes: bytes) -> str | None:
    """Return embedded pi_smiles metadata from PNG bytes when present."""
    from io import BytesIO

    from PIL import Image

    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception:
        return None
    text = getattr(img, "text", None) or {}
    raw = text.get(PNG_SMILES_KEY) or img.info.get(PNG_SMILES_KEY)
    if not raw:
        return None
    try:
        return canonicalize_smiles(str(raw))
    except SmilesDescriptorError:
        return None
