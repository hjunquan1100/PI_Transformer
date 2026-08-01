"""
generate.py

Generate PI repeat-unit candidates conditioned on a target Tg.

Examples:
    python src/generate.py --tg_target 300 --n_samples 50

    # Python API:
    from generate import load_model, generate_batch
    model, vocab, inv_vocab, stats = load_model("checkpoints/best_model.pt")
    output = generate_batch(model, tg_target=300, n_samples=50, ...)
"""

import os
import argparse
import json
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    raise ImportError(f"Missing dependencies: {e}")

from model     import PIGeneratorModel
from paths import DEFAULT_GENERATOR_CKPT
from validator import filter_generated_results, print_report


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a generator checkpoint and return (model, vocab, inv_vocab, stats)."""
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg  = ckpt["model_config"]
    model = PIGeneratorModel(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    vocab     = ckpt["vocab"]
    inv_vocab = ckpt["inv_vocab"]
    stats     = ckpt["stats"]

    print(f"Model loaded: {ckpt_path}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model, vocab, inv_vocab, stats


def resolve_ckpt_path(ckpt_path: str) -> str:
    """Resolve best_model.pt to the newest best_model_valloss_*.pt file."""
    if os.path.exists(ckpt_path):
        return ckpt_path

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    ckpt_name = os.path.basename(ckpt_path)
    if ckpt_name != "best_model.pt" or not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    candidates = []
    for root, _, files in os.walk(ckpt_dir):
        for name in files:
            if name.startswith("best_model_valloss_") and name.endswith(".pt"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def tg_to_tensors(tg_target, stats, device):
    """Convert a target Tg value to normalized and binned tensors."""
    tg_norm = (tg_target - stats["tg_mean"]) / stats["tg_std"]
    tg_norm_t = torch.tensor([tg_norm], dtype=torch.float, device=device)

    bins      = np.array(stats["tg_bins"])
    n_bins    = stats["n_bins"]
    bin_id    = int(np.digitize(tg_target, bins)) - 1
    bin_id    = max(0, min(bin_id, n_bins - 1))
    tg_bin_t  = torch.tensor([bin_id], dtype=torch.long, device=device)

    return tg_norm_t, tg_bin_t


def decode_tokens(tokens: list, inv_vocab: dict, bos_id: int, eos_id: int) -> str:
    """Decode token ids to a SELFIES string."""
    out = []
    for t in tokens:
        if t == bos_id:
            continue
        if t == eos_id:
            break
        out.append(inv_vocab.get(t, ""))
    return "".join(out)


def selfies_to_valid_smiles(selfies_str: str):
    """Convert SELFIES to canonical SMILES and return None if RDKit rejects it."""
    try:
        smiles = sf.decoder(selfies_str)
        mol    = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def make_result_id(index: int) -> str:
    """Create a stable candidate ID."""
    return f"GEN_{index:04d}"


def prepare_run_paths(out_path: str) -> tuple[str, str, str, str]:
    """
    Create a timestamped output directory.

    Returns (run_dir, passed_csv_path, all_csv_path, structures_dir).
    """
    out_dir = os.path.dirname(out_path) or "results"
    stem = os.path.splitext(os.path.basename(out_path))[0] or "generated"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{stem}_{timestamp}")
    structures_dir = os.path.join(run_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)
    return (
        run_dir,
        os.path.join(run_dir, f"{stem}.csv"),
        os.path.join(run_dir, f"{stem}_all.csv"),
        structures_dir,
    )


def draw_structure_images(results: list, structures_dir: str, verbose: bool = True) -> int:
    """Render generated structures to PNG files."""
    saved = 0
    for row in results:
        smiles = row.get("smiles", "")
        mol_id = row.get("id", f"GEN_{saved+1:04d}")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        image_path = os.path.join(structures_dir, f"{mol_id}.png")
        Draw.MolToFile(
            mol,
            image_path,
            size=(1400, 320),
            kekulize=False,
            legend=mol_id,
        )
        saved += 1
    if verbose:
        print(f"Structure images saved: {saved} images -> {structures_dir}")
    return saved


def save_results_csv(path: str, rows: list):
    """Save generated rows to CSV."""
    fieldnames = [
        "id",
        "smiles",
        "selfies",
        "tg_target",
        "valid",
        "imide_ring_count",
        "mol_weight",
        "num_heavy_atoms",
        "num_rings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def sample_top_p(
    model,
    tg_norm_t:   torch.Tensor,
    tg_bin_t:    torch.Tensor,
    vocab:       dict,
    inv_vocab:   dict,
    temperature: float = 1.0,
    top_p:       float = 0.9,
    max_len:     int   = 300,
    device:      str   = "cpu",
):
    """Run top-p sampling and return (smiles_or_none, selfies_str)."""
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]

    tokens = [bos_id]
    for _ in range(max_len):
        inp    = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(tg_norm_t, tg_bin_t, inp)[:, -1, :]  # (1, vocab_size)

        # Temperature scaling.
        logits = logits / max(temperature, 1e-6)
        probs  = F.softmax(logits, dim=-1)[0]                 # (vocab_size,)

        # Top-p filtering
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest token set whose cumulative probability reaches top_p.
        mask   = (cumsum - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-8)

        # sampling
        next_tok = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item()
        tokens.append(next_tok)
        if next_tok == eos_id:
            break

    selfies_str = decode_tokens(tokens, inv_vocab, bos_id, eos_id)
    smiles      = selfies_to_valid_smiles(selfies_str)
    return smiles, selfies_str


@torch.no_grad()
def beam_search(
    model,
    tg_norm_t:  torch.Tensor,
    tg_bin_t:   torch.Tensor,
    vocab:      dict,
    inv_vocab:  dict,
    beam_width: int = 5,
    max_len:    int = 300,
    device:     str = "cpu",
):
    """Run beam search and return (smiles_or_none, selfies_str, score)."""
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]

    # beams: list of (log_prob, token_list, finished)
    beams     = [(0.0, [bos_id], False)]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        any_active = False

        for score, toks, finished in beams:
            if finished:
                completed.append((score / max(len(toks) - 1, 1), toks))
                continue
            any_active = True
            inp    = torch.tensor([toks], dtype=torch.long, device=device)
            logits = model(tg_norm_t, tg_bin_t, inp)[:, -1, :]
            log_p  = F.log_softmax(logits, dim=-1)[0]
            topv, topi = log_p.topk(beam_width)

            for v, i in zip(topv.tolist(), topi.tolist()):
                new_score = score + v
                new_toks  = toks + [i]
                is_done   = (i == eos_id)
                if is_done:
                    length_norm = len(new_toks) - 1
                    completed.append((new_score / max(length_norm, 1), new_toks))
                else:
                    all_candidates.append((new_score, new_toks, False))

        if not any_active or not all_candidates:
            break
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    # Add unfinished beams as completed candidates.
    for score, toks, _ in beams:
        length_norm = len(toks) - 1
        completed.append((score / max(length_norm, 1), toks))

    if not completed:
        return None, "", float("-inf")

    best_score, best_toks = max(completed, key=lambda x: x[0])
    selfies_str = decode_tokens(best_toks, inv_vocab, bos_id, eos_id)
    smiles      = selfies_to_valid_smiles(selfies_str)
    return smiles, selfies_str, best_score


def generate_batch(
    model,
    tg_target:   float,
    vocab:       dict,
    inv_vocab:   dict,
    stats:       dict,
    n_samples:   int   = 100,
    temperature: float = 1.0,
    top_p:       float = 0.9,
    max_len:     int   = 300,
    device:      str   = "cpu",
    use_beam:    bool  = False,
    beam_width:  int   = 5,
    validate:    bool  = True,
    strict:      bool  = True,
    verbose:     bool  = True,
):
    """
    Generate candidates and optionally filter them by PI plausibility.

    Returns a dictionary:
        {
            "all_results": [...],
            "passed_results": [...],
            "valid_count": int,
            "pass_rate": float,
        }
    """
    tg_norm_t, tg_bin_t = tg_to_tensors(tg_target, stats, device)
    results = []
    valid_count = 0

    if verbose:
        print(f"\nGenerating {n_samples} candidates for target Tg = {tg_target} deg C")

    for i in range(n_samples):
        if use_beam:
            smiles, selfies_str, _ = beam_search(
                model, tg_norm_t, tg_bin_t, vocab, inv_vocab,
                beam_width=beam_width, max_len=max_len, device=device,
            )
        else:
            smiles, selfies_str = sample_top_p(
                model, tg_norm_t, tg_bin_t, vocab, inv_vocab,
                temperature=temperature, top_p=top_p,
                max_len=max_len, device=device,
            )

        is_valid = smiles is not None
        if is_valid:
            valid_count += 1

        results.append({
            "id":        make_result_id(i + 1),
            "smiles":    smiles if is_valid else "",
            "selfies":   selfies_str,
            "tg_target": tg_target,
            "valid":     is_valid,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}]  valid: {valid_count/(i+1)*100:.1f}%")

    if verbose:
        print(f"\nRDKit valid: {valid_count}/{n_samples} = {valid_count/n_samples*100:.1f}%")

    passed_results = results
    if validate:
        if verbose:
            print("\nRunning PI plausibility validation")
        passed_results, all_vr = filter_generated_results(results, strict=strict)
        if verbose:
            print_report(all_vr)
            print(f"PI-plausible candidates: {len(passed_results)} / {valid_count} RDKit-valid structures")

    return {
        "all_results":    results,
        "passed_results": passed_results,
        "valid_count":    valid_count,
        "pass_rate":      round(len(passed_results) / max(valid_count, 1), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI inverse design")
    parser.add_argument("--ckpt",        default=str(DEFAULT_GENERATOR_CKPT))
    parser.add_argument("--tg_target",   type=float, default=300.0, help="Target Tg in deg C")
    parser.add_argument("--n_samples",   type=int,   default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--max_len",     type=int,   default=300)
    parser.add_argument("--use_beam",    action="store_true", help="Use beam search instead of top-p sampling")
    parser.add_argument("--beam_width",  type=int,   default=5)
    parser.add_argument("--out",         default="results/generated.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, inv_vocab, stats = load_model(args.ckpt, device=device)

    output = generate_batch(
        model        = model,
        tg_target    = args.tg_target,
        vocab        = vocab,
        inv_vocab    = inv_vocab,
        stats        = stats,
        n_samples    = args.n_samples,
        temperature  = args.temperature,
        top_p        = args.top_p,
        max_len      = args.max_len,
        device       = device,
        use_beam     = args.use_beam,
        beam_width   = args.beam_width,
        validate     = True,
        strict       = True,
    )

    run_dir, passed_csv_path, all_csv_path, structures_dir = prepare_run_paths(args.out)

    passed = output["passed_results"]
    all_results = output["all_results"]
    save_results_csv(all_csv_path, all_results)
    save_results_csv(passed_csv_path, passed)

    draw_structure_images(all_results, structures_dir)

    summary = {
        "tg_target": args.tg_target,
        "n_samples": args.n_samples,
        "rdkit_valid_count": output["valid_count"],
        "pi_pass_count": len(passed),
        "pi_pass_rate": output["pass_rate"],
        "passed_csv": passed_csv_path,
        "all_csv": all_csv_path,
        "structures_dir": structures_dir,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResult directory: {run_dir}")
    print(f"PI-plausible results saved: {passed_csv_path}  ({len(passed)} rows)")
    print(f"All generated results saved: {all_csv_path}  ({len(all_results)} rows)")
