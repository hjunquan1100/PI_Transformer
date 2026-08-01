#!/usr/bin/env python3
"""Plot real-vs-generated PI chemical-space and structure diagnostics.

The prepare stage requires RDKit. It canonicalizes and deduplicates the original
1000 x 3 generation batch, samples 1000 generated structures, and saves Morgan
fingerprints, descriptors, and nearest-neighbor similarities. The plot stage
requires umap-learn and produces three separate publication figures.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
TRAIN_XLSX = ROOT / "data/raw/data/data.xlsx"
BATCH_DIR = ROOT / "results/batch_generation/batch_targets_20260327_102735"
OUT_DIR = ROOT / "results/paper_figures/real_vs_generated_structure"

REAL_COLOR = "#A7ADB4"
GENERATED_COLOR = "#2374AB"
INK = "#17202A"
MUTED = "#66717E"
def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8491")
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#3E4854", labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save_figure(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
        print("Saved figure:", path)
    plt.close(fig)


def read_training_table(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name="PI structures and SMILES")


def prepare_data(out_dir: Path, train_table: Path, sample_size: int, real_sample_size: int, seed: int):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    RDLogger.DisableLog("rdApp.*")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def mol_and_canonical(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None
        return mol, Chem.MolToSmiles(mol, canonical=True)

    def fingerprint(mol):
        bit_vector = fp_generator.GetFingerprint(mol)
        array = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(bit_vector, array)
        return bit_vector, array

    def imide_ring_count(mol):
        patterns = [
            Chem.MolFromSmarts("[#6]1(=O)[#7][#6](=O)[#6][#6]1"),
            Chem.MolFromSmarts("[#6]1(=O)[#7][#6](=O)c[c,n]1"),
            Chem.MolFromSmarts("O=C1[#7]C(=O)[#6,#7][#6,#7]1"),
        ]
        matches = set()
        for pattern in patterns:
            for match in mol.GetSubstructMatches(pattern):
                matches.add(frozenset(match))
        return len(matches)

    def descriptor_record(smiles):
        mol, canonical = mol_and_canonical(smiles)
        if mol is None:
            return None
        return {
            "canonical_smiles": canonical,
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "rotatable_bond_count": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "imide_ring_count": int(imide_ring_count(mol)),
        }

    raw_training = read_training_table(train_table)
    real_records = []
    seen_real = set()
    for smiles in raw_training["SMILES-Repeating unit"].dropna().astype(str):
        record = descriptor_record(smiles)
        if record is not None and record["canonical_smiles"] not in seen_real:
            seen_real.add(record["canonical_smiles"])
            real_records.append(record)
    real = pd.DataFrame(real_records)
    real["group"] = "Real polyimides"
    if len(real) < real_sample_size:
        raise ValueError(f"Only {len(real)} unique real structures; cannot sample {real_sample_size}")
    real_display = real.sample(n=real_sample_size, random_state=seed, replace=False).reset_index(drop=True)

    generated_frames = []
    source_counts = {}
    for target in (200, 300, 400):
        path = BATCH_DIR / f"tg_{target}/generated_passed.csv"
        frame = pd.read_csv(path)
        source_counts[str(target)] = int(len(frame))
        frame["source_target_tg_c"] = target
        generated_frames.append(frame[["smiles", "source_target_tg_c"]])
    generated_raw = pd.concat(generated_frames, ignore_index=True)

    generated_records = []
    canonical_targets = {}
    for row in generated_raw.itertuples(index=False):
        record = descriptor_record(row.smiles)
        if record is None:
            continue
        canonical = record["canonical_smiles"]
        canonical_targets.setdefault(canonical, set()).add(int(row.source_target_tg_c))
        if len(canonical_targets[canonical]) == 1:
            generated_records.append(record)
    generated_pool = pd.DataFrame(generated_records).drop_duplicates("canonical_smiles").reset_index(drop=True)
    generated_pool["source_targets"] = generated_pool["canonical_smiles"].map(
        lambda value: ",".join(str(v) for v in sorted(canonical_targets[value]))
    )
    if len(generated_pool) < sample_size:
        raise ValueError(f"Only {len(generated_pool)} unique generated structures; cannot sample {sample_size}")
    generated = generated_pool.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)
    generated["group"] = "Generated polyimides"

    combined = pd.concat([real_display, generated], ignore_index=True)
    bit_vectors = []
    fingerprint_arrays = []
    for smiles in combined["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        bit_vector, array = fingerprint(mol)
        bit_vectors.append(bit_vector)
        fingerprint_arrays.append(array)
    fingerprint_matrix = np.stack(fingerprint_arrays)

    real_bit_vectors = []
    for smiles in real["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        bit_vector, _ = fingerprint(mol)
        real_bit_vectors.append(bit_vector)
    generated_bit_vectors = bit_vectors[len(real_display):]
    nearest_similarities = []
    nearest_indices = []
    for generated_fp in generated_bit_vectors:
        similarities = DataStructs.BulkTanimotoSimilarity(generated_fp, real_bit_vectors)
        nearest_index = int(np.argmax(similarities))
        nearest_indices.append(nearest_index)
        nearest_similarities.append(float(similarities[nearest_index]))
    generated["nearest_real_tanimoto"] = nearest_similarities
    generated["nearest_real_smiles"] = [real.iloc[index]["canonical_smiles"] for index in nearest_indices]
    generated["exact_training_match"] = generated["canonical_smiles"].isin(set(real["canonical_smiles"]))

    real.to_csv(out_dir / "real_polyimides_descriptors.csv", index=False)
    real_display.to_csv(out_dir / "real_sample_1000_descriptors.csv", index=False)
    generated.to_csv(out_dir / "generated_sample_1000_descriptors_and_similarity.csv", index=False)
    combined[["canonical_smiles", "group"]].to_csv(out_dir / "umap_input_metadata.csv", index=False)
    np.savez_compressed(out_dir / "umap_input_fingerprints.npz", fingerprints=fingerprint_matrix)

    similarity = generated["nearest_real_tanimoto"].to_numpy(float)
    summary = {
        "source_batch": str(BATCH_DIR.relative_to(ROOT)),
        "source_passed_counts": source_counts,
        "generated_unique_union": int(len(generated_pool)),
        "generated_sample_size": int(len(generated)),
        "real_unique_size": int(len(real)),
        "real_display_sample_size": int(len(real_display)),
        "nearest_neighbor_real_reference_size": int(len(real)),
        "random_seed": seed,
        "fingerprint": {"type": "Morgan bit vector", "radius": 2, "n_bits": 2048},
        "nearest_real_tanimoto": {
            "mean": float(np.mean(similarity)),
            "median": float(np.median(similarity)),
            "q1": float(np.percentile(similarity, 25)),
            "q3": float(np.percentile(similarity, 75)),
            "fraction_below_0_5": float(np.mean(similarity < 0.5)),
            "fraction_0_5_to_0_8": float(np.mean((similarity >= 0.5) & (similarity < 0.8))),
            "fraction_at_least_0_8": float(np.mean(similarity >= 0.8)),
            "fraction_at_least_0_9": float(np.mean(similarity >= 0.9)),
            "exact_training_match_fraction": float(generated["exact_training_match"].mean()),
        },
    }
    with (out_dir / "real_vs_generated_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def plot_umap(out_dir: Path, seed: int):
    embedding_path = out_dir / "umap_embedding.csv"
    if embedding_path.exists():
        metadata = pd.read_csv(embedding_path)
    else:
        import umap

        metadata = pd.read_csv(out_dir / "umap_input_metadata.csv")
        fingerprints = np.load(out_dir / "umap_input_fingerprints.npz")["fingerprints"]
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.15,
            n_components=2,
            metric="jaccard",
            random_state=seed,
            n_jobs=1,
            low_memory=True,
        )
        embedding = reducer.fit_transform(fingerprints)
        metadata["umap_1"] = embedding[:, 0]
        metadata["umap_2"] = embedding[:, 1]
        metadata.to_csv(embedding_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    real_mask = metadata["group"] == "Real polyimides"
    generated_mask = ~real_mask
    ax.scatter(
        metadata.loc[real_mask, "umap_1"], metadata.loc[real_mask, "umap_2"],
        s=12, color=REAL_COLOR, alpha=0.38, linewidth=0, rasterized=True,
        label=f"Real polyimides (n={int(real_mask.sum())})", zorder=1,
    )
    ax.scatter(
        metadata.loc[generated_mask, "umap_1"], metadata.loc[generated_mask, "umap_2"],
        s=23, color=GENERATED_COLOR, alpha=0.68, linewidth=0, rasterized=True,
        label=f"Generated polyimides (n={int(generated_mask.sum())})", zorder=2,
    )
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(frameon=True, facecolor="white", edgecolor="#C8CFD7", fontsize=10, markerscale=1.35)
    style_axes(ax)
    save_figure(fig, out_dir, "figure_a_real_generated_umap")


def kde_curve(values, grid):
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < 2:
        return np.zeros_like(grid)
    return gaussian_kde(values)(grid)


def plot_descriptors(out_dir: Path):
    real = pd.read_csv(out_dir / "real_sample_1000_descriptors.csv")
    generated = pd.read_csv(out_dir / "generated_sample_1000_descriptors_and_similarity.csv")
    descriptors = [
        ("molecular_weight", "Molecular weight (g mol$^{-1}$)"),
        ("heavy_atom_count", "Heavy atom count"),
        ("ring_count", "Ring count"),
        ("imide_ring_count", "Imide ring count"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.3), layout="constrained")
    for ax, (column, label) in zip(axes.flat, descriptors):
        real_values = real[column].to_numpy(float)
        generated_values = generated[column].to_numpy(float)
        lower = min(real_values.min(), generated_values.min())
        upper = max(real_values.max(), generated_values.max())
        padding = max((upper - lower) * 0.06, 0.25)
        grid = np.linspace(lower - padding, upper + padding, 500)
        real_density = kde_curve(real_values, grid)
        generated_density = kde_curve(generated_values, grid)

        ax.fill_between(grid, real_density, color=REAL_COLOR, alpha=0.28)
        ax.plot(grid, real_density, color="#7B828A", linewidth=1.8)
        ax.fill_between(grid, generated_density, color=GENERATED_COLOR, alpha=0.23)
        ax.plot(grid, generated_density, color=GENERATED_COLOR, linewidth=2.0)
        ax.axvline(np.median(real_values), color="#7B828A", linestyle="--", linewidth=1.1)
        ax.axvline(np.median(generated_values), color=GENERATED_COLOR, linestyle="--", linewidth=1.1)
        ax.set_xlabel(label)
        ax.set_ylabel("Probability density")
        ax.text(
            0.98, 0.94,
            f"Median: {np.median(real_values):.1f} vs {np.median(generated_values):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D6DC", "alpha": 0.88},
        )
        style_axes(ax)

    handles = [
        Patch(facecolor=REAL_COLOR, edgecolor="#7B828A", alpha=0.45, label=f"Real (n={len(real)})"),
        Patch(facecolor=GENERATED_COLOR, edgecolor=GENERATED_COLOR, alpha=0.35, label=f"Generated (n={len(generated)})"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.995), frameon=False, ncol=2, fontsize=10)
    save_figure(fig, out_dir, "figure_b_structural_descriptor_distributions")


def plot_nearest_neighbor(out_dir: Path):
    generated = pd.read_csv(out_dir / "generated_sample_1000_descriptors_and_similarity.csv")
    values = generated["nearest_real_tanimoto"].to_numpy(float)
    bins = np.linspace(0, 1, 31)
    grid = np.linspace(max(0, values.min() - 0.05), 1.0, 500)
    density = kde_curve(values, grid)

    fig, ax = plt.subplots(figsize=(7.2, 5.1), layout="constrained")
    weights = np.ones_like(values) * 100.0 / len(values)
    ax.hist(values, bins=bins, weights=weights, color=GENERATED_COLOR, alpha=0.72,
            edgecolor="white", linewidth=0.7, label="Generated structures")
    bin_width = bins[1] - bins[0]
    ax.plot(grid, density * 100 * bin_width, color=INK, linewidth=2.0, label="KDE")
    median = float(np.median(values))
    ax.axvline(median, color="#C23B33", linestyle="--", linewidth=1.8, label=f"Median = {median:.3f}")
    ax.axvspan(0.9, 1.0, color="#F4C7C3", alpha=0.28, label="High similarity (>=0.9)")
    ax.set_xlim(0, 1.01)
    ax.set_xlabel("Maximum Tanimoto similarity to real polyimides")
    ax.set_ylabel("Generated samples per bin (%)")
    stats = (
        f"Q1–Q3: {np.percentile(values, 25):.3f}–{np.percentile(values, 75):.3f}\n"
        f"Similarity >=0.9: {np.mean(values >= 0.9) * 100:.1f}%\n"
        f"Exact training matches: {generated['exact_training_match'].mean() * 100:.1f}%"
    )
    ax.text(0.975, 0.965, stats, transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#C8CFD7", "alpha": 0.92})
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    style_axes(ax)
    save_figure(fig, out_dir, "figure_c_nearest_neighbor_tanimoto")


def plot_data(out_dir: Path, seed: int):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_umap(out_dir, seed)
    plot_descriptors(out_dir)
    plot_nearest_neighbor(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plot", "all"))
    parser.add_argument("--train-table", default=str(TRAIN_XLSX))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--real-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_table = Path(args.train_table).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.stage in ("prepare", "all"):
        prepare_data(out_dir, train_table, args.sample_size, args.real_sample_size, args.seed)
    if args.stage in ("plot", "all"):
        plot_data(out_dir, args.seed)


if __name__ == "__main__":
    main()
