#!/usr/bin/env python3
"""Prepare data and plot target-conditioned PI generation behavior.

Stage ``prepare`` requires RDKit and creates canonical unique candidate data,
forward-model Tg predictions, and balanced Morgan-fingerprint input for UMAP.
Stage ``plot`` requires umap-learn and creates the two publication figures.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/paper_figures/target_conditioned_generation"
TRAIN_XLSX = ROOT / "data/raw/data/data.xlsx"
FORWARD_MODEL = ROOT / "checkpoints/forward_20260318_232230/forward_model_best_r2_0.8535.pkl"
CANDIDATE_FILES = {
    200: ROOT / "results/jp_src3_generated_200_20260423_170808/jp_src3_generated_200.csv",
    300: ROOT / "results/jp_src3_generated_300_20260423_171102/jp_src3_generated_300.csv",
    400: ROOT / "results/jp_src3_generated_400_20260423_164610/jp_src3_generated_400.csv",
}

COLORS = {200: "#2878B5", 300: "#E68600", 400: "#C23B33"}
TRAIN_COLOR = "#A6ADB4"
INK = "#17202A"
MUTED = "#66717E"
GRID = "#D9DEE5"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8491")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#3E4854", labelsize=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save_figure(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
        print("Saved figure:", path)
    plt.close(fig)


def prepare_data(out_dir: Path, sample_per_target: int, seed: int, train_table: Path):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import ConvertToNumpyArray

    RDLogger.DisableLog("rdApp.*")
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = generator.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        return arr

    if train_table.suffix.lower() == ".csv":
        training = pd.read_csv(train_table)
    else:
        training = pd.read_excel(train_table, sheet_name="PI structures and SMILES")
    training = training[["SMILES-Repeating unit", "Tg( deg C)"]].dropna().copy()
    training["canonical_smiles"] = training["SMILES-Repeating unit"].map(canonicalize)
    training = training.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)
    training = training.rename(columns={"Tg( deg C)": "experimental_tg_c"})

    candidate_frames = []
    source_counts = {}
    for target, path in CANDIDATE_FILES.items():
        frame = pd.read_csv(path)
        source_counts[str(target)] = {
            "source_rows": int(len(frame)),
            "source_unique_raw_smiles": int(frame["smiles"].nunique()),
        }
        frame = frame.loc[frame["valid"].astype(bool)].copy()
        frame["canonical_smiles"] = frame["smiles"].map(canonicalize)
        frame = frame.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)
        frame["target_tg_c"] = float(target)
        frame["source_file"] = str(path.relative_to(ROOT))
        source_counts[str(target)]["canonical_unique_candidates"] = int(len(frame))
        candidate_frames.append(frame)
    candidates = pd.concat(candidate_frames, ignore_index=True)

    with FORWARD_MODEL.open("rb") as handle:
        forward_model = pickle.load(handle)
    candidate_fps = np.stack([fingerprint(smiles) for smiles in candidates["canonical_smiles"]])
    candidates["predicted_tg_c"] = forward_model.predict(candidate_fps.astype(np.int8))
    candidates["signed_target_error_c"] = candidates["predicted_tg_c"] - candidates["target_tg_c"]
    candidates["absolute_target_error_c"] = np.abs(candidates["signed_target_error_c"])
    candidates.to_csv(out_dir / "unique_candidates_with_predicted_tg.csv", index=False)

    training_set = set(training["canonical_smiles"])
    candidates["seen_in_training"] = candidates["canonical_smiles"].isin(training_set)
    rng = np.random.default_rng(seed)
    sampled = []
    for target in sorted(CANDIDATE_FILES):
        group = candidates.loc[candidates["target_tg_c"] == target]
        n_take = min(sample_per_target, len(group))
        indices = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        sampled.append(candidates.loc[indices, ["canonical_smiles", "target_tg_c", "predicted_tg_c"]].copy())
    sampled_candidates = pd.concat(sampled, ignore_index=True)

    umap_metadata = pd.DataFrame(
        {
            "canonical_smiles": training["canonical_smiles"],
            "group": "Training set",
            "target_tg_c": np.nan,
            "predicted_tg_c": np.nan,
        }
    )
    for target in sorted(CANDIDATE_FILES):
        group = sampled_candidates.loc[sampled_candidates["target_tg_c"] == target].copy()
        group["group"] = f"Target {target} °C"
        umap_metadata = pd.concat(
            [umap_metadata, group[["canonical_smiles", "group", "target_tg_c", "predicted_tg_c"]]],
            ignore_index=True,
        )
    umap_fps = np.stack([fingerprint(smiles) for smiles in umap_metadata["canonical_smiles"]])
    umap_metadata.to_csv(out_dir / "umap_input_metadata.csv", index=False)
    np.savez_compressed(out_dir / "umap_input_fingerprints.npz", fingerprints=umap_fps)

    target_metrics = {}
    for target in sorted(CANDIDATE_FILES):
        values = candidates.loc[candidates["target_tg_c"] == target]
        errors = values["absolute_target_error_c"].to_numpy(float)
        target_metrics[str(target)] = {
            "n_unique": int(len(values)),
            "predicted_tg_mean_c": float(values["predicted_tg_c"].mean()),
            "predicted_tg_median_c": float(values["predicted_tg_c"].median()),
            "predicted_tg_std_c": float(values["predicted_tg_c"].std(ddof=0)),
            "target_mae_c": float(np.mean(errors)),
            "hit_at_20": float(np.mean(errors <= 20)),
            "hit_at_30": float(np.mean(errors <= 30)),
            "seen_in_training_fraction": float(values["seen_in_training"].mean()),
        }

    overlap = {}
    target_sets = {
        target: set(candidates.loc[candidates["target_tg_c"] == target, "canonical_smiles"])
        for target in sorted(CANDIDATE_FILES)
    }
    for left, right in ((200, 300), (200, 400), (300, 400)):
        intersection = len(target_sets[left] & target_sets[right])
        union = len(target_sets[left] | target_sets[right])
        overlap[f"{left}_{right}"] = {
            "intersection": intersection,
            "jaccard": float(intersection / union) if union else 0.0,
        }

    summary = {
        "training_unique": int(len(training)),
        "candidate_sources": source_counts,
        "target_metrics": target_metrics,
        "cross_target_overlap": overlap,
        "forward_model": str(FORWARD_MODEL.relative_to(ROOT)),
        "forward_model_cv": {"best_fold_r2": 0.8535, "five_fold_mean_r2": 0.8058, "five_fold_mean_mae_c": 21.779},
        "umap_sampling": {"sample_per_target": sample_per_target, "random_seed": seed, "training_used": int(len(training))},
        "fingerprint": {"type": "Morgan bit vector", "radius": 2, "n_bits": 2048},
    }
    with (out_dir / "target_conditioned_generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def plot_umap(out_dir: Path, seed: int):
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
    metadata.to_csv(out_dir / "umap_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    train = metadata["group"] == "Training set"
    for target in sorted(CANDIDATE_FILES):
        mask = metadata["target_tg_c"] == target
        ax.scatter(
            metadata.loc[mask, "umap_1"], metadata.loc[mask, "umap_2"],
            s=17, color=COLORS[target], alpha=0.48, linewidth=0, rasterized=True, zorder=2,
            label=f"Target {target} °C (n={int(mask.sum())})",
        )
    ax.scatter(
        metadata.loc[train, "umap_1"], metadata.loc[train, "umap_2"],
        s=13, color=TRAIN_COLOR, alpha=0.55, linewidth=0, rasterized=True, zorder=3,
        label=f"Training set (n={int(train.sum())})",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical-space distribution of generated candidates", loc="left", fontsize=14, weight="bold", color=INK, pad=30)
    ax.text(0.0, 1.012, "Morgan fingerprints (radius 2, 2048 bit); Jaccard distance",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    order = [len(handles) - 1] + list(range(len(handles) - 1))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="best",
              frameon=True, facecolor="white", edgecolor="#C8CFD7", fontsize=8.5, markerscale=1.4)
    style_axes(ax)
    save_figure(fig, out_dir, "figure3a_umap_structure_space")


def plot_property_distribution(out_dir: Path):
    candidates = pd.read_csv(out_dir / "unique_candidates_with_predicted_tg.csv")
    targets = sorted(CANDIDATE_FILES)
    distributions = [
        candidates.loc[candidates["target_tg_c"] == target, "predicted_tg_c"].to_numpy(float)
        for target in targets
    ]
    all_values = np.concatenate(distributions)
    lower = math.floor(np.min(all_values) / 25) * 25
    upper = max(425, math.ceil(np.max(all_values) / 25) * 25)

    fig, ax = plt.subplots(figsize=(7.4, 5.7), layout="constrained")
    parts = ax.violinplot(distributions, positions=np.arange(len(targets)), widths=0.72,
                          showmeans=False, showmedians=False, showextrema=False, bw_method=0.25)
    for body, target in zip(parts["bodies"], targets):
        body.set_facecolor(COLORS[target])
        body.set_edgecolor(COLORS[target])
        body.set_alpha(0.42)

    rng = np.random.default_rng(2026)
    for position, (target, values) in enumerate(zip(targets, distributions)):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.vlines(position, q1, q3, color=INK, linewidth=7, zorder=4)
        ax.scatter([position], [median], color="white", edgecolor=INK, linewidth=1.0, s=38, zorder=5)
        ax.hlines(target, position - 0.37, position + 0.37, color=COLORS[target], linestyle="--", linewidth=2.0, zorder=3)
        sample_size = min(500, len(values))
        sampled = rng.choice(values, sample_size, replace=False)
        jitter = rng.normal(position, 0.055, sample_size)
        ax.scatter(jitter, sampled, color=COLORS[target], s=7, alpha=0.14, linewidth=0, rasterized=True, zorder=2)

        errors = np.abs(values - target)
        metric_text = (
            f"n={len(values):,}\n"
            f"Target MAE={np.mean(errors):.1f} °C\n"
            f"Hit@20={np.mean(errors <= 20) * 100:.1f}%"
        )
        ax.text(position, lower + 0.035 * (upper - lower), metric_text, ha="center", va="bottom", fontsize=8.2, color=INK,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C8CFD7", "alpha": 0.88})

    ax.set_xlim(-0.65, len(targets) - 0.35)
    ax.set_ylim(lower, upper)
    ax.set_xticks(np.arange(len(targets)), [f"Target {target} °C" for target in targets])
    ax.set_ylabel("Forward-model predicted $T_g$ (°C)")
    ax.set_xlabel("Generation condition")
    ax.set_title("Predicted property distributions by target condition", loc="left", fontsize=14, weight="bold", color=INK, pad=30)
    ax.text(0.0, 1.012, "Unique PI-rule-passing candidates; dashed segments mark conditioning targets",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    legend = [
        Line2D([0], [0], color=INK, linewidth=7, marker="o", markerfacecolor="white", markeredgecolor=INK,
               label="Interquartile range and median"),
        Line2D([0], [0], color=MUTED, linestyle="--", linewidth=2, label="Conditioning target"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8.5)
    style_axes(ax)
    save_figure(fig, out_dir, "figure3b_predicted_tg_distribution")


def plot_data(out_dir: Path, seed: int):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    plot_umap(out_dir, seed)
    plot_property_distribution(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plot", "all"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--train-table", default=str(TRAIN_XLSX))
    parser.add_argument("--sample-per-target", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    train_table = Path(args.train_table).expanduser().resolve()
    if args.stage in ("prepare", "all"):
        prepare_data(out_dir, args.sample_per_target, args.seed, train_table)
    if args.stage in ("plot", "all"):
        plot_data(out_dir, args.seed)


if __name__ == "__main__":
    main()
