#!/usr/bin/env python3
"""Generate five publication-ready PI Tg prediction figures.

The saved downstream checkpoint belongs to one CV fold. This script reconstructs
that fold's preprocessing, evaluates the untouched 80-sample holdout set, saves
the point predictions, and draws five figures. Use --plots-only to redraw from
an existing prediction CSV without importing PyTorch/Transformers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/config/p26_runs/kfold_m128_head15.yaml"
DEFAULT_CKPT = ROOT / "ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt"
DEFAULT_HOLDOUT = ROOT / "data/PI_Tg_10066_holdout80_p1_m128.csv"
DEFAULT_PREDICTIONS = ROOT / "experiments/results/holdout80_fold5_predictions.csv"
DEFAULT_BASELINE = ROOT / "experiments/results/finetune_baseline_mlm_seq.json"
DEFAULT_PROPOSED = ROOT / "experiments/results/p26_run_kfold_m128_head15.json"
DEFAULT_OUT_DIR = ROOT / "experiments/figures/pi_tg_prediction"

BLUE = "#176B87"
ORANGE = "#D97706"
RED = "#B42318"
GREEN = "#2F855A"
INK = "#17202A"
GRID = "#D9DEE5"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def kfold_indices(n: int, k: int, random_state: int = 1):
    """Match sklearn KFold(n_splits=k, shuffle=True, random_state=...)."""
    # sklearn RandomState shuffling is stable and avoids importing sklearn in
    # the inference-only environment.
    indices = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    current = 0
    splits = []
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        splits.append((train_idx, val_idx))
        current = stop
    return splits


def standardize(train: np.ndarray, other: np.ndarray):
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (other - mean) / scale, mean, scale


def run_inference(config_path: Path, checkpoint_path: Path, holdout_path: Path, output_path: Path):
    import torch
    import torch.nn as nn
    from transformers import RobertaModel

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from PolymerSmilesTokenization import PolymerSmilesTokenizer

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_fold_zero = int(checkpoint.get("fold", -1))
    if checkpoint_fold_zero < 0:
        raise ValueError("Checkpoint has no fold metadata; fold preprocessing cannot be reconstructed")

    train_pool = pd.read_csv(resolve_path(cfg["train_file"]))
    holdout = pd.read_csv(holdout_path)
    desc_cols = list(cfg["descriptor_cols"])
    missing = [column for column in desc_cols if column not in holdout.columns]
    if missing:
        raise ValueError("Holdout is missing descriptor columns: %s" % ", ".join(missing[:8]))

    splits = kfold_indices(len(train_pool), int(cfg["k"]), random_state=1)
    fold_train_idx, _ = splits[checkpoint_fold_zero]
    fold_train = train_pool.iloc[fold_train_idx].reset_index(drop=True)

    holdout_desc, _, _ = standardize(
        fold_train[desc_cols].to_numpy(dtype=np.float64),
        holdout[desc_cols].to_numpy(dtype=np.float64),
    )
    fold_train_y = fold_train["value"].to_numpy(dtype=np.float64)
    y_mean = float(np.mean(fold_train_y))
    y_scale = float(np.std(fold_train_y, ddof=0))

    model_dir = resolve_path(cfg["model_path"])
    tokenizer = PolymerSmilesTokenizer.from_pretrained(str(model_dir), max_len=int(cfg["blocksize"]))
    if cfg.get("add_vocab_flag"):
        supplemental = pd.read_csv(resolve_path(cfg["vocab_sup_file"]), header=None).values.flatten().tolist()
        tokenizer.add_tokens(supplemental)

    backbone = RobertaModel.from_pretrained(str(model_dir))
    backbone.config.hidden_dropout_prob = float(cfg["hidden_dropout_prob"])
    backbone.config.attention_probs_dropout_prob = float(cfg["attention_probs_dropout_prob"])
    backbone.resize_token_embeddings(len(tokenizer))

    class RegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.PretrainedModel = backbone
            hidden = int(backbone.config.hidden_size)
            hidden_head = max(1, int(round(hidden * float(cfg.get("reg_head_hidden_mult", 1.0)))))
            self.Regressor = nn.Sequential(
                nn.Dropout(float(cfg["drop_rate"])),
                nn.Linear(hidden + len(desc_cols), hidden_head),
                nn.SiLU(),
                nn.Linear(hidden_head, 1),
            )

        def forward(self, input_ids, attention_mask, descriptors):
            encoded = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = encoded.last_hidden_state[:, 0, :]
            return self.Regressor(torch.cat((cls_embedding, descriptors), dim=-1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    predictions = []
    batch_size = min(16, int(cfg.get("batch_size", 16)))
    smiles = holdout["smiles"].astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            stop = min(start + batch_size, len(smiles))
            encoded = tokenizer(
                smiles[start:stop],
                add_special_tokens=True,
                max_length=int(cfg["blocksize"]),
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            descriptor_tensor = torch.tensor(holdout_desc[start:stop], dtype=torch.float32, device=device)
            standardized = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                descriptor_tensor,
            ).squeeze(-1)
            predictions.extend((standardized.cpu().numpy() * y_scale + y_mean).tolist())

    result = pd.DataFrame(
        {
            "sample_index": np.arange(len(holdout)),
            "smiles": holdout["smiles"],
            "experimental_tg_c": holdout["value"].astype(float),
            "predicted_tg_c": np.asarray(predictions, dtype=float),
        }
    )
    result["residual_c"] = result["predicted_tg_c"] - result["experimental_tg_c"]
    result["absolute_error_c"] = np.abs(result["residual_c"])
    result["checkpoint_fold_1based"] = checkpoint_fold_zero + 1
    result["checkpoint_epoch_1based"] = int(checkpoint.get("epoch", -1)) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print("Saved predictions:", output_path)
    return result


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true, y_pred):
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denominator)


def bootstrap_intervals(y_true, y_pred, n_boot=5000, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = {"mae": [], "rmse": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        values["mae"].append(mae(yt, yp))
        values["rmse"].append(rmse(yt, yp))
        if np.var(yt) > 0:
            values["r2"].append(r2(yt, yp))
    return {key: np.percentile(val, [2.5, 97.5]).tolist() for key, val in values.items()}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8491")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#3E4854", labelsize=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save_figure(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
        print("Saved figure:", path)
    plt.close(fig)


def figure_parity(data, metrics, intervals, out_dir):
    yt = data["experimental_tg_c"].to_numpy(float)
    yp = data["predicted_tg_c"].to_numpy(float)
    ae = np.abs(yp - yt)
    low = math.floor((min(yt.min(), yp.min()) - 20) / 50) * 50
    high = math.ceil((max(yt.max(), yp.max()) + 20) / 50) * 50
    line = np.linspace(low, high, 400)

    fig, ax = plt.subplots(figsize=(6.4, 5.8), layout="constrained")
    ax.fill_between(line, line - 40, line + 40, color="#FDE8D2", alpha=0.55, label=r"$\pm$40 °C")
    ax.fill_between(line, line - 20, line + 20, color="#CFE8DF", alpha=0.75, label=r"$\pm$20 °C")
    ax.plot(line, line, color=INK, linewidth=1.6, label="Ideal: y = x")
    points = ax.scatter(yt, yp, c=ae, cmap="viridis", s=42, edgecolor="white", linewidth=0.5, zorder=3)
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Absolute error (°C)", fontsize=9)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Experimental $T_g$ (°C)", ylabel="Predicted $T_g$ (°C)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Independent holdout parity", loc="left", fontsize=14, color=INK, weight="bold", pad=30)
    ax.text(0.0, 1.012, "Fold-5 checkpoint; all 80 untouched holdout samples", transform=ax.transAxes, fontsize=9, color="#5F6B78")
    text = (
        f"RMSE  {metrics['rmse']:.2f} °C  [{intervals['rmse'][0]:.2f}, {intervals['rmse'][1]:.2f}]\n"
        f"MAE    {metrics['mae']:.2f} °C  [{intervals['mae'][0]:.2f}, {intervals['mae'][1]:.2f}]\n"
        f"$R^2$       {metrics['r2']:.3f}      [{intervals['r2'][0]:.3f}, {intervals['r2'][1]:.3f}]"
    )
    ax.text(0.035, 0.965, text, transform=ax.transAxes, va="top", fontsize=9, family="monospace", color=INK,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#B8C1CC", "alpha": 0.94})
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    style_axes(ax)
    save_figure(fig, out_dir, "01_holdout_parity")


def figure_residuals(data, metrics, out_dir):
    yt = data["experimental_tg_c"].to_numpy(float)
    residual = data["residual_c"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    ax.axhspan(-metrics["rmse"], metrics["rmse"], color="#E6F0ED", alpha=0.8, label=r"$\pm$ holdout RMSE")
    ax.axhline(0, color=INK, linewidth=1.5)
    ax.scatter(yt, residual, color=BLUE, alpha=0.78, s=40, edgecolor="white", linewidth=0.5, zorder=3)

    quantiles = np.quantile(yt, np.linspace(0, 1, 7))
    centers, means = [], []
    for i in range(len(quantiles) - 1):
        mask = (yt >= quantiles[i]) & (yt <= quantiles[i + 1] if i == len(quantiles) - 2 else yt < quantiles[i + 1])
        if np.any(mask):
            centers.append(float(np.mean(yt[mask])))
            means.append(float(np.mean(residual[mask])))
    ax.plot(centers, means, color=RED, marker="o", markersize=4, linewidth=1.8, label="Binned mean residual")
    ax.set(xlabel="Experimental $T_g$ (°C)", ylabel="Residual: predicted - experimental (°C)")
    ax.set_title("Residual diagnostics", loc="left", fontsize=14, color=INK, weight="bold", pad=30)
    bias = float(np.mean(residual))
    ax.text(0.01, 1.012, f"Mean bias = {bias:+.2f} °C; random scatter around zero is preferred", transform=ax.transAxes,
            fontsize=9, color="#5F6B78")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    style_axes(ax)
    save_figure(fig, out_dir, "02_holdout_residuals")


def figure_error_ecdf(data, out_dir):
    errors = np.sort(data["absolute_error_c"].to_numpy(float))
    cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    ax.step(errors, cumulative, where="post", color=BLUE, linewidth=2.5)
    threshold_colors = {20: GREEN, 30: ORANGE, 50: RED}
    for threshold, color in threshold_colors.items():
        pct = float(np.mean(errors <= threshold) * 100)
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.2)
        ax.scatter([threshold], [pct], color=color, s=45, zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(threshold + 2, pct - 2, f"{pct:.1f}% within {threshold} °C", color=color, fontsize=9, va="top")
    ax.set(xlabel="Absolute prediction error (°C)", ylabel="Cumulative samples (%)", ylim=(0, 102))
    ax.set_title("Absolute-error cumulative distribution", loc="left", fontsize=14, color=INK, weight="bold", pad=30)
    ax.text(0.01, 1.012, f"Median = {np.median(errors):.2f} °C; 90th percentile = {np.percentile(errors, 90):.2f} °C",
            transform=ax.transAxes, fontsize=9, color="#5F6B78")
    style_axes(ax)
    save_figure(fig, out_dir, "03_holdout_absolute_error_ecdf")


def figure_cv_comparison(baseline_path, proposed_path, out_dir):
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with proposed_path.open("r", encoding="utf-8") as handle:
        proposed = json.load(handle)
    base_rmse = np.asarray(baseline["per_fold_val_rmse"], dtype=float)
    prop_rmse = np.asarray(proposed["per_fold_val_rmse"], dtype=float)
    base_r2 = np.asarray(baseline["per_fold_val_r2"], dtype=float)
    prop_r2 = np.asarray(proposed["per_fold_val_r2"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), layout="constrained")
    x = np.array([0, 1])
    for fold in range(len(base_rmse)):
        axes[0].plot(x, [base_rmse[fold], prop_rmse[fold]], color="#AAB2BD", linewidth=1.1, zorder=1)
        axes[0].scatter(x, [base_rmse[fold], prop_rmse[fold]], c=[ORANGE, BLUE], s=48, zorder=2, edgecolor="white", linewidth=0.5)
        axes[1].plot(x, [base_r2[fold], prop_r2[fold]], color="#AAB2BD", linewidth=1.1, zorder=1)
        axes[1].scatter(x, [base_r2[fold], prop_r2[fold]], c=[ORANGE, BLUE], s=48, zorder=2, edgecolor="white", linewidth=0.5)
    for ax, base_values, prop_values, ylabel in (
        (axes[0], base_rmse, prop_rmse, "Validation RMSE (°C)"),
        (axes[1], base_r2, prop_r2, "Validation $R^2$"),
    ):
        ax.errorbar([0, 1], [base_values.mean(), prop_values.mean()], yerr=[base_values.std(), prop_values.std()],
                    fmt="D", color=INK, markersize=5, capsize=4, linewidth=1.3, zorder=4, label="Mean ± SD")
        ax.set_xticks(x, ["MLM + SMILES\nbaseline", "Proposed\nmodel"])
        ax.set_ylabel(ylabel)
        style_axes(ax)
    relative = (base_rmse.mean() - prop_rmse.mean()) / base_rmse.mean() * 100
    axes[0].text(0.5, 0.04, f"RMSE reduced by {relative:.1f}%\nImproved in {np.sum(prop_rmse < base_rmse)}/{len(base_rmse)} folds",
                 transform=axes[0].transAxes, ha="center", fontsize=9, color=GREEN)
    axes[1].text(0.5, 0.04, f"Mean $R^2$: {base_r2.mean():.3f} → {prop_r2.mean():.3f}",
                 transform=axes[1].transAxes, ha="center", fontsize=9, color=GREEN)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Five-fold cross-validation comparison", x=0.02, ha="left", fontsize=14, color=INK, weight="bold")
    save_figure(fig, out_dir, "04_five_fold_model_comparison")


def figure_tg_ranges(data, out_dir):
    boundaries = [-np.inf, 200, 300, 400, np.inf]
    labels = ["<200", "200–300", "300–400", ">=400"]
    groups = pd.cut(data["experimental_tg_c"], bins=boundaries, labels=labels, right=False)
    errors = data["absolute_error_c"].to_numpy(float)
    grouped = [errors[(groups == label).to_numpy()] for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
    boxes = ax.boxplot(grouped, labels=labels, widths=0.56, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "linewidth": 1.5},
                       whiskerprops={"color": "#74808D"}, capprops={"color": "#74808D"})
    colors = ["#A8DADC", "#7FC8A9", "#F6BD60", "#E5989B"]
    rng = np.random.default_rng(17)
    for index, (box, values, color) in enumerate(zip(boxes["boxes"], grouped, colors), start=1):
        box.set_facecolor(color)
        box.set_alpha(0.58)
        jitter = rng.normal(index, 0.055, len(values))
        ax.scatter(jitter, values, color=INK, alpha=0.55, s=21, edgecolor="none", zorder=3)
        if len(values):
            ax.text(index, 42, f"MAE {np.mean(values):.1f}\nn={len(values)}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set(xlabel="Experimental $T_g$ range (°C)", ylabel="Absolute prediction error (°C)")
    ax.set_title("Prediction error across temperature ranges", loc="left", fontsize=14, color=INK, weight="bold", pad=30)
    ax.text(0.01, 1.012, "Boxes show median and interquartile range; dots are individual holdout samples",
            transform=ax.transAxes, fontsize=9, color="#5F6B78")
    style_axes(ax)
    save_figure(fig, out_dir, "05_holdout_error_by_tg_range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE))
    parser.add_argument("--proposed-results", default=str(DEFAULT_PROPOSED))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    holdout_path = resolve_path(args.holdout)
    predictions_path = resolve_path(args.predictions)
    out_dir = resolve_path(args.out_dir)

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

    if args.plots_only:
        data = pd.read_csv(predictions_path)
    else:
        data = run_inference(config_path, checkpoint_path, holdout_path, predictions_path)

    y_true = data["experimental_tg_c"].to_numpy(float)
    y_pred = data["predicted_tg_c"].to_numpy(float)
    metrics = {"n": len(data), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
    intervals = bootstrap_intervals(y_true, y_pred)
    summary = {
        **metrics,
        "bootstrap_95_ci": intervals,
        "mean_bias_c": float(np.mean(y_pred - y_true)),
        "median_absolute_error_c": float(np.median(np.abs(y_pred - y_true))),
        "pct_within_20c": float(np.mean(np.abs(y_pred - y_true) <= 20) * 100),
        "pct_within_30c": float(np.mean(np.abs(y_pred - y_true) <= 30) * 100),
        "pct_within_50c": float(np.mean(np.abs(y_pred - y_true) <= 50) * 100),
        "evaluation_scope": "untouched holdout80 evaluated with saved fold-5 checkpoint",
        "prediction_csv": str(predictions_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "holdout_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    figure_parity(data, metrics, intervals, out_dir)
    figure_residuals(data, metrics, out_dir)
    figure_error_ecdf(data, out_dir)
    figure_cv_comparison(resolve_path(args.baseline_results), resolve_path(args.proposed_results), out_dir)
    figure_tg_ranges(data, out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
