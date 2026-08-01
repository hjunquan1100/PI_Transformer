# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import random

import pandas as pd
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaModel, get_linear_schedule_with_warmup

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import ContrastivePolymerPairDataset


def load_smiles_for_contrastive(pretrain_csv, max_contrastive_samples, subsample_seed):
   

    path = Path(pretrain_csv)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    cap = max_contrastive_samples
    use_cap = cap is not None and str(cap).strip() != "" and int(cap) > 0
    if not use_cap:
        raw = pd.read_csv(path, header=None).values
        lst = [str(raw[i, 0]).strip() for i in range(len(raw))]
        return lst, len(lst), False

    cap = int(cap)
    rng = random.Random(int(subsample_seed))
    reservoir = []
    i = -1
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            i += 1
            s = str(row[0]).strip()
            if i < cap:
                reservoir.append(s)
            else:
                j = rng.randint(0, i)
                if j < cap:
                    reservoir[j] = s
    n_seen = i + 1
    if n_seen == 0:
        raise ValueError("No data rows in %s" % pretrain_csv)
    did_limit = n_seen > cap
    return reservoir, n_seen, did_limit


def _suppress_rdkit_warnings():
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def info_nce_loss(z1, z2, temperature):   
    logits = torch.mm(z1, z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)


class ContrastivePolymerModel(nn.Module):
    def __init__(self, encoder: RobertaModel, proj_dim: int):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        z = self.projection(h)
        return F.normalize(z, dim=-1)


def freeze_roberta_bottom_layers(roberta: RobertaModel, n_layers: int):
    if n_layers <= 0:
        return
    for i in range(min(n_layers, len(roberta.encoder.layer))):
        for p in roberta.encoder.layer[i].parameters():
            p.requires_grad = False


def main(cfg):
    _suppress_rdkit_warnings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tok_path = cfg.get("tokenizer_path", "roberta-base")
    tokenizer = PolymerSmilesTokenizer.from_pretrained(tok_path, max_len=cfg["blocksize"])
    if cfg.get("add_vocab_flag"):
        vocab_sup = pd.read_csv(cfg["vocab_sup_file"], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    smiles_list, n_csv_total, did_cap = load_smiles_for_contrastive(
        cfg["pretrain_csv"],
        cfg.get("max_contrastive_samples"),
        cfg.get("subsample_seed", 42),
    )
    cap_cfg = cfg.get("max_contrastive_samples")
    if cap_cfg is not None and str(cap_cfg).strip() != "" and int(cap_cfg) > 0:
        cap_i = int(cap_cfg)
        if did_cap:
            print(
                "max_contrastive_samples=%d: reservoir sample over %d valid CSV rows -> %d used (seed=%s)"
                % (cap_i, n_csv_total, len(smiles_list), cfg.get("subsample_seed", 42))
            )
        else:
            print(
                "max_contrastive_samples=%d: CSV has %d valid rows (<= cap), using all (seed=%s)"
                % (cap_i, n_csv_total, cfg.get("subsample_seed", 42))
            )
    else:
        print("Contrastive corpus size (full CSV load): %d rows - first epoch may take very long." % n_csv_total)

    ds = ContrastivePolymerPairDataset(
        smiles_list,
        tokenizer,
        cfg["blocksize"],
        aug_indicator=cfg.get("aug_indicator") if cfg.get("aug_indicator") is not None else 4,
    )
    nw = int(cfg.get("num_workers", 0))
    _dl_kw = dict(
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    if nw > 0:
        _dl_kw["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        _dl_kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    loader = DataLoader(ds, **_dl_kw)
    print("Batches per epoch (drop_last=True): %d, batch_size=%d" % (len(loader), int(cfg["batch_size"])))

    mlm_path = cfg["mlm_checkpoint"]
    encoder = RobertaModel.from_pretrained(mlm_path)
    encoder.resize_token_embeddings(len(tokenizer))
    encoder.config.hidden_dropout_prob = float(cfg.get("hidden_dropout_prob", 0.1))
    encoder.config.attention_probs_dropout_prob = float(cfg.get("attention_probs_dropout_prob", 0.1))

    freeze_roberta_bottom_layers(encoder, int(cfg.get("freeze_encoder_layers", 0)))

    model = ContrastivePolymerModel(encoder, int(cfg["proj_dim"])).to(device)
    # FP64  GPU , empty; please FP32 (default). FP64  use_fp64: true
    if cfg.get("use_fp64", False):
        model = model.double()

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.05)))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    tau = float(cfg.get("temperature", 0.1))
    model.train()
    step = 0
    epoch_mean_losses = []
    n_epochs = int(cfg["epochs"])
    for epoch in range(n_epochs):
        ep_loss = 0.0
        pbar = tqdm(loader, desc="Contrastive epoch %d/%d" % (epoch + 1, n_epochs), leave=True)
        for batch in pbar:
            ids1 = batch["input_ids_1"].to(device, non_blocking=True).long()
            m1 = batch["attention_mask_1"].to(device, non_blocking=True).long()
            ids2 = batch["input_ids_2"].to(device, non_blocking=True).long()
            m2 = batch["attention_mask_2"].to(device, non_blocking=True).long()
            z1 = model(ids1, m1)
            z2 = model(ids2, m2)
            loss = info_nce_loss(z1, z2, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            li = float(loss.item())
            ep_loss += li
            step += 1
            pbar.set_postfix(loss="%.4f" % li, refresh=False)
        mean_ep = ep_loss / max(len(loader), 1)
        epoch_mean_losses.append(mean_ep)
        print("epoch %s/%s contrastive loss (mean batch) = %.6f" % (epoch + 1, cfg["epochs"], mean_ep))

    out_dir = cfg["save_path"]
    os.makedirs(out_dir, exist_ok=True)
    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved encoder + tokenizer to", os.path.abspath(out_dir))

    mj = cfg.get("metrics_json_path")
    if mj:
        p = Path(mj)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment_name": cfg.get("experiment_name", ""),
                    "mlm_checkpoint": cfg.get("mlm_checkpoint"),
                    "save_path": cfg.get("save_path"),
                    "pretrain_csv": cfg.get("pretrain_csv"),
                    "n_rows_in_csv": int(n_csv_total),
                    "n_rows_in_csv_valid_scanned": int(n_csv_total),
                    "n_samples_used_contrastive": int(len(smiles_list)),
                    "reservoir_limited_to_cap": bool(did_cap),
                    "max_contrastive_samples_config": cfg.get("max_contrastive_samples"),
                    "subsample_seed": cfg.get("subsample_seed"),
                    "epochs": int(cfg["epochs"]),
                    "batch_size": int(cfg["batch_size"]),
                    "temperature": float(cfg.get("temperature", 0.1)),
                    "proj_dim": int(cfg["proj_dim"]),
                    "epoch_mean_contrastive_loss": epoch_mean_losses,
                    "final_mean_contrastive_loss": float(epoch_mean_losses[-1]) if epoch_mean_losses else None,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("Saved contrastive metrics JSON ->", p.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Contrastive pretrain after MLM")
    ap.add_argument(
        "--config",
        type=str,
        default="config_contrastive.yaml",
        help=" YAML (default config_contrastive.yaml;  experiments/config/contrastive_full.yaml)",
    )
    cli = ap.parse_args()
    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print("config file:", cli.config)
    print(cfg)
    main(cfg)
