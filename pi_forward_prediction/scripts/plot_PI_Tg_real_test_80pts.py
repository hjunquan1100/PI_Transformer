
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import RobertaModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "results" / "PI_Tg_real_test80_predicted_vs_true.png"
CONFIG_PATH = ROOT / "config_finetune.yaml"
N_PLOT = 80


def draw_panel(ax, true, pred, title, mae_d: float, rmse_d: float, r2_d: float):
    ax.scatter(
        true,
        pred,
        c="#4A90E2",
        alpha=0.64,
        s=28,
        edgecolors="none",
        rasterized=True,
    )
    lim0, lim1 = 0, 700
    ax.plot([lim0, lim1], [lim0, lim1], "r--", linewidth=2.0, label="y = x")
    ax.set_xlim(lim0, lim1)
    ax.set_ylim(lim0, lim1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="-", alpha=0.32)
    ax.legend(loc="lower right", fontsize=9)

    text = (
        f"MAE = {mae_d:.4f}\n"
        f"RMSE = {rmse_d:.4f}\n"
        f"$R^2$ = {r2_d:.4f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="wheat",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        ),
    )


def subsample_even_spread(y_true: np.ndarray, n: int) -> np.ndarray:
    """ Tg  n items,  ()."""
    order = np.argsort(y_true)
    m = len(order)
    if m <= n:
        return order
    pos = np.linspace(0, m - 1, n)
    idx_in_sorted = np.clip(np.round(pos).astype(np.int64), 0, m - 1)
    return order[idx_in_sorted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="KFold validationfold (1-based), default 1;  k fold best checkpoint k fold",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_PATH),
        help="output PNG path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML config path relative to this directory; empty uses the packaged default",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint .pt path; empty uses best_model_path from the YAML config",
    )
    args = parser.parse_args()

    if args.config.strip():
        cfg_path = ROOT / args.config
        if not cfg_path.is_file():
            raise SystemExit(f"--config file does not exist: {cfg_path}")
    else:
        cfg_path = CONFIG_PATH if CONFIG_PATH.is_file() else ROOT / "results" / "config_finetune_PI_Tg_snapshot.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    from PolymerSmilesTokenization import PolymerSmilesTokenizer

    model_path = ROOT / str(cfg["model_path"])
    if str(args.ckpt).strip():
        _qp = Path(str(args.ckpt).strip())
        best_path = _qp if _qp.is_absolute() else (ROOT / _qp)
    else:
        best_path = ROOT / str(cfg["best_model_path"])
    train_csv = ROOT / str(cfg["train_file"])

    pretrained = RobertaModel.from_pretrained(str(model_path))
    pretrained.config.hidden_dropout_prob = cfg["hidden_dropout_prob"]
    pretrained.config.attention_probs_dropout_prob = cfg["attention_probs_dropout_prob"]

    tok_path = ROOT / "roberta-base"
    tokenizer = PolymerSmilesTokenizer.from_pretrained(str(tok_path), max_len=cfg["blocksize"])
    if cfg.get("add_vocab_flag"):
        vocab_sup = pd.read_csv(ROOT / str(cfg["vocab_sup_file"]), header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    data = pd.read_csv(train_csv)
    k = int(cfg["k"])
    fold_1based = int(args.fold)
    if not (1 <= fold_1based <= k):
        raise SystemExit(f"--fold must be in 1..{k} between, received {fold_1based}")
    fold_zero = fold_1based - 1

    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    splits = list(kf.split(np.arange(data.shape[0])))
    train_idx, val_idx = splits[fold_zero]
    train_data = data.loc[train_idx, :].reset_index(drop=True)
    val_data = data.loc[val_idx, :].reset_index(drop=True)

    from dataset import Downstream_Dataset
    from Downstream import (
        DownstreamRegression,
        assert_descriptor_columns_present,
        descriptor_cols_from_config,
        scale_descriptor_block,
    )

    desc_cols = descriptor_cols_from_config(cfg)
    use_fusion = bool(cfg.get("use_descriptor_fusion")) and desc_cols
    if use_fusion:
        assert_descriptor_columns_present(train_data, desc_cols)
        assert_descriptor_columns_present(val_data, desc_cols)
        train_data, val_data = scale_descriptor_block(train_data, val_data, desc_cols)

    scaler = StandardScaler()
    train_data = train_data.copy()
    val_data = val_data.copy()
    train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
    val_data.iloc[:, 1] = scaler.transform(val_data.iloc[:, 1].values.reshape(-1, 1))

    val_ds = Downstream_Dataset(
        val_data,
        tokenizer,
        cfg["blocksize"],
        descriptor_cols=desc_cols if use_fusion else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    n_desc = len(desc_cols) if use_fusion else 0
    model = DownstreamRegression(
        pretrained,
        tokenizer,
        drop_rate=cfg["drop_rate"],
        use_descriptor_fusion=use_fusion,
        n_desc=n_desc,
        descriptor_ablation=cfg.get("descriptor_ablation", "none"),
        descriptor_ablation_column=int(cfg.get("descriptor_ablation_column", 0)),
        reg_head_hidden_mult=float(cfg.get("reg_head_hidden_mult", 1.0)),
    ).to(device)
    model = model.float()
    try:
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(best_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            desc = batch["desc"].to(device).float() if use_fusion else None
            out = model(input_ids, attention_mask, desc).float()
            out = torch.from_numpy(scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)))
            prop_u = torch.from_numpy(scaler.inverse_transform(prop.cpu().numpy().reshape(-1, 1)))
            preds.append(out.flatten().numpy())
            trues.append(prop_u.flatten().numpy())

    y_pred = np.concatenate(preds).astype(np.float64)
    y_true = np.concatenate(trues).astype(np.float64)

    sel = subsample_even_spread(y_true, N_PLOT)
    yt = y_true[sel]
    yp = y_pred[sel]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.8, 6.5), dpi=150)
    title = (
        f"Predicted vs True (PI $T_g$, °C)\n"
        f"n={N_PLOT} pts evenly spread on fold {fold_1based} val "
        f"(train CSV n={data.shape[0]}, ckpt {best_path.name})"
    )
    draw_panel(ax, yt, yp, title, mae, rmse, r2)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print("Saved", out_path.resolve())
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} on {N_PLOT} points")
    print(f"True range [{yt.min():.1f}, {yt.max():.1f}]")


if __name__ == "__main__":
    main()
