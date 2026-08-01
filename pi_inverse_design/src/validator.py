"""Plausibility checks for generated polyimide repeat units."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

@dataclass
class ValidationResult:
    smiles: str
    passed: bool = False
    rdkit_valid: bool = False
    has_imide_ring: bool = False
    has_imide_carbonyl: bool = False
    imide_ring_count: int = 0
    mol_weight: Optional[float] = None
    num_heavy_atoms: int = 0
    num_rings: int = 0
    mw_ok: bool = False
    atoms_ok: bool = False
    has_unstable_groups: bool = False
    unstable_groups: list = field(default_factory=list)
    fail_reasons: list = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{status}  {self.smiles[:70]}"]
        lines.append(f"  [L1] RDKit: {self.rdkit_valid}")
        if self.rdkit_valid:
            lines.append(
                f"  [L2] imide ring: {self.has_imide_ring} "
                f"(x{self.imide_ring_count})  imide carbonyl: {self.has_imide_carbonyl}"
            )
            lines.append(f"  [L3] MW={self.mol_weight:.1f}  heavy atoms={self.num_heavy_atoms}  unstable groups={self.has_unstable_groups}")
        if self.fail_reasons:
            lines.append(f"  failures: {'; '.join(self.fail_reasons)}")
        return "\n".join(lines)


# SMARTS patterns for PI plausibility checks.
_IMIDE_CARBONYL  = Chem.MolFromSmarts("[#6](=O)-[#7]-[#6](=O)")
_IMIDE_RING_SAT  = Chem.MolFromSmarts("[#6]1(=O)[#7][#6](=O)[#6][#6]1")
_IMIDE_RING_ARO  = Chem.MolFromSmarts("[#6]1(=O)[#7][#6](=O)c[c,n]1")
_IMIDE_RING_GEN  = Chem.MolFromSmarts("O=C1[#7]C(=O)[#6,#7][#6,#7]1")

_UNSTABLE = [
    (Chem.MolFromSmarts("[N+](=O)[O-]"), "nitro group"),
    (Chem.MolFromSmarts("[N]=[N+]=[N-]"), "azide group"),
    (Chem.MolFromSmarts("[O-][O+]"), "peroxide-like motif"),
    (Chem.MolFromSmarts("O=N-O"), "nitroso hydroxyl motif"),
    (Chem.MolFromSmarts("[Hg,Pb,As,Cd,Cr]"), "heavy-metal atom"),
    (Chem.MolFromSmarts("C1OO1"), "dioxirane-like ring"),
    (Chem.MolFromSmarts("[#7]~[#7]~[#7]"), "nitrogen chain"),
    (Chem.MolFromSmarts("C(=S)N"), "thioamide group"),
]
_UNSTABLE = [(p, d) for p, d in _UNSTABLE if p is not None]

MW_MIN, MW_MAX = 150.0, 2000.0
HEAVY_MIN, HEAVY_MAX = 10, 150


def validate_smiles(
    smiles: str,
    require_imide_ring: bool = True,
    require_imide_carbonyl: bool = True,
    check_mw: bool = True,
    check_unstable: bool = True,
) -> ValidationResult:
    res = ValidationResult(smiles=smiles)

    # Layer 1: RDKit parse and canonicalization.
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        res.fail_reasons.append("RDKit could not parse the SMILES")
        return res
    res.rdkit_valid = True
    res.smiles = Chem.MolToSmiles(mol, canonical=True)

    # Layer 2: imide substructure.
    if _IMIDE_CARBONYL and mol.HasSubstructMatch(_IMIDE_CARBONYL):
        res.has_imide_carbonyl = True
    ring_matches = set()
    for pat in [_IMIDE_RING_SAT, _IMIDE_RING_ARO, _IMIDE_RING_GEN]:
        if pat is None:
            continue
        for m in mol.GetSubstructMatches(pat):
            ring_matches.add(frozenset(m))
    if ring_matches:
        res.has_imide_ring = True
        res.imide_ring_count = len(ring_matches)
        res.has_imide_carbonyl = True

    if require_imide_ring and not res.has_imide_ring:
        res.fail_reasons.append("missing imide ring")
    elif require_imide_carbonyl and not res.has_imide_carbonyl:
        res.fail_reasons.append("missing imide -C(=O)-N-C(=O)- motif")

    # Layer 3: size and unstable-group filters.
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()
    res.mol_weight = round(mw, 2)
    res.num_heavy_atoms = heavy
    res.num_rings = rdMolDescriptors.CalcNumRings(mol)
    res.mw_ok = MW_MIN <= mw <= MW_MAX
    res.atoms_ok = HEAVY_MIN <= heavy <= HEAVY_MAX

    if check_mw and not res.mw_ok:
        res.fail_reasons.append(f"MW={mw:.1f} [{MW_MIN},{MW_MAX}]")
    if check_mw and not res.atoms_ok:
        res.fail_reasons.append(f"heavy atoms={heavy} [{HEAVY_MIN},{HEAVY_MAX}]")

    if check_unstable:
        for pat, desc in _UNSTABLE:
            if mol.HasSubstructMatch(pat):
                res.has_unstable_groups = True
                res.unstable_groups.append(desc)
        if res.has_unstable_groups:
            res.fail_reasons.append(f"unstable groups: {','.join(res.unstable_groups)}")

    res.passed = len(res.fail_reasons) == 0
    return res


def validate_batch(smiles_list: list, verbose: bool = False, **kwargs) -> list:
    results = []
    for smi in smiles_list:
        vr = validate_smiles(smi, **kwargs)
        results.append(vr)
        if verbose:
            print(vr.summary())
    return results


def print_report(vr_list: list):
    total   = len(vr_list)
    passed  = sum(1 for r in vr_list if r.passed)
    l1      = sum(1 for r in vr_list if r.rdkit_valid)
    l2_ring = sum(1 for r in vr_list if r.has_imide_ring)
    l2_skel = sum(1 for r in vr_list if r.has_imide_carbonyl)
    l3_mw   = sum(1 for r in vr_list if r.mw_ok)
    l3_stab = sum(1 for r in vr_list if r.rdkit_valid and not r.has_unstable_groups)
    mws     = [r.mol_weight for r in vr_list if r.mol_weight]

    print("\n" + "=" * 55)
    print("Polyimide Structure Validation")
    print("=" * 55)
    print(f"Total candidates: {total}")
    print(f"passed (PASS):   {passed}/{total}  ({passed/total*100:.1f}%)")
    print()
    print(f"[L1] RDKit valid:       {l1}/{total}  ({l1/total*100:.1f}%)")
    if l1 > 0:
        print(f"[L2] imide ring:       {l2_ring}/{l1}  ({l2_ring/l1*100:.1f}%)")
        print(f"[L2] imide carbonyl:   {l2_skel}/{l1}  ({l2_skel/l1*100:.1f}%)")
        print(f"[L3] MW plausible:     {l3_mw}/{l1}  ({l3_mw/l1*100:.1f}%)")
        print(f"[L3] stable groups:    {l3_stab}/{l1}  ({l3_stab/l1*100:.1f}%)")
    if mws:
        print(f"\nMW statistics: mean={np.mean(mws):.1f}  min={np.min(mws):.1f}  max={np.max(mws):.1f}")
    print("=" * 55)


def filter_generated_results(generate_results: list, strict: bool = True) -> tuple:
    """
    Filter generated structures by PI plausibility.

    Returns (passed_rows, validation_results).
    """
    valid_smiles = list({
        r["smiles"] for r in generate_results
        if r.get("valid") and r.get("smiles")
    })
    all_vr = validate_batch(
        valid_smiles,
        require_imide_ring     = strict,
        require_imide_carbonyl = True,
    )
    vr_map = {vr.smiles: vr for vr in all_vr}

    passed = []
    for r in generate_results:
        smi = r.get("smiles", "")
        if not smi:
            continue
        try:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except Exception:
            canon = smi
        vr = vr_map.get(canon) or vr_map.get(smi)
        if vr and vr.passed:
            enriched = dict(r)
            enriched["imide_ring_count"] = vr.imide_ring_count
            enriched["mol_weight"]       = vr.mol_weight
            enriched["num_heavy_atoms"]  = vr.num_heavy_atoms
            enriched["num_rings"]        = vr.num_rings
            passed.append(enriched)

    return passed, all_vr
