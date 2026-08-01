import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Downstream_Dataset, DataAugmentation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from copy import deepcopy

from prep_PI_Tg_descriptors import DESC_COLS as DEFAULT_PI_TG_DESCRIPTOR_COLS

np.random.seed(seed=1)


def descriptor_cols_from_config(finetune_config):
    if not finetune_config.get("use_descriptor_fusion", False):
        return None
    cols = finetune_config.get("descriptor_cols")
    if cols is None:
        return list(DEFAULT_PI_TG_DESCRIPTOR_COLS)
    return list(cols)


def assert_descriptor_columns_present(df, desc_cols):
    for c in desc_cols:
        if c not in df.columns:
            raise ValueError("use_descriptor_fusion=True but column %r missing in CSV" % c)


def scale_descriptor_block(train_df, test_df, desc_cols):
    scaler_d = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[desc_cols] = scaler_d.fit_transform(train_df[desc_cols].values)
    test_df[desc_cols] = scaler_d.transform(test_df[desc_cols].values)
    return train_df, test_df


def ensure_checkpoint_parent_dirs(finetune_config):
    for key in ("save_path", "best_model_path"):
        p = finetune_config.get(key)
        if p:
            Path(p).expanduser().parent.mkdir(parents=True, exist_ok=True)


def dataloader_kw(finetune_config):
    nw = int(finetune_config.get("num_workers", 0))
    return {
        "num_workers": nw,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": bool(nw > 0),
    }


def val_selection_metric_from_config(finetune_config):
    m = (finetune_config.get("val_selection_metric") or "r2").lower().strip()
    if m not in ("r2", "rmse"):
        raise ValueError("val_selection_metric must be 'r2' or 'rmse', got %r" % m)
    return m


def validation_epoch_is_improvement(finetune_config, r2_val, mse_val, best_r2, best_mse, has_best):
    """Return whether the current epoch improves the configured validation metric."""
    m = val_selection_metric_from_config(finetune_config)
    if m == "rmse":
        if not has_best:
            return True
        return float(mse_val) < float(best_mse)
    if not has_best:
        return True
    return float(r2_val) > float(best_r2)


def build_cv_split_indices(finetune_config, data):
    """Return [(train_idx, val_idx), ...] using KFold or StratifiedKFold."""
    k = int(finetune_config["k"])
    n = data.shape[0]
    idx = np.arange(n)
    cv_mode = (finetune_config.get("cv_mode") or "kfold").lower().strip()
    if cv_mode == "kfold":
        sp = KFold(n_splits=k, shuffle=True, random_state=1)
        return list(sp.split(idx))
    if cv_mode == "stratified_regression":
        y = data.iloc[:, 1].values.astype(float)
        q = int(finetune_config.get("stratify_n_bins", 10))
        q_eff = max(2, min(q, max(2, n // max(k, 1))))
        try:
            bins = pd.qcut(y, q=q_eff, labels=False, duplicates="drop")
            if isinstance(bins, pd.Series):
                bins = bins.to_numpy()
            bins = np.asarray(bins, dtype=np.int64)
            if np.any(np.isnan(bins.astype(float))):
                bins = np.nan_to_num(bins, nan=0).astype(np.int64)
        except (ValueError, TypeError) as e:
            print("stratified_regression qcut failed (%s); falling back to kfold" % e)
            sp = KFold(n_splits=k, shuffle=True, random_state=1)
            return list(sp.split(idx))
        _, counts = np.unique(bins, return_counts=True)
        if np.any(counts < k):
            print(
                "stratified_regression: some bin count < k; falling back to kfold "
                "(bins min count=%s)" % int(counts.min())
            )
            sp = KFold(n_splits=k, shuffle=True, random_state=1)
            return list(sp.split(idx))
        sp = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
        return list(sp.split(idx, bins))
    raise ValueError("cv_mode must be 'kfold' or 'stratified_regression', got %r" % cv_mode)


def build_training_loss(finetune_config):
    """Build the configured training loss."""
    lt = (finetune_config.get("loss_type") or "mse").lower().strip()
    if lt == "quantile":
        return nn.MSELoss()
    if lt == "huber":
        beta = float(finetune_config.get("huber_beta", 1.0))
        return nn.SmoothL1Loss(beta=beta)
    return nn.MSELoss()


def compute_train_sample_weights(raw_y_celsius: np.ndarray, mode: str):
    """Return optional sample weights from raw Tg values; mean weight is normalized to 1."""
    m = (mode or "none").lower().strip()
    if m in ("", "none"):
        return None
    if m == "inv_var_tg":
        mu = float(np.mean(raw_y_celsius))
        sig = float(np.std(raw_y_celsius)) + 1e-8
        z = (raw_y_celsius.astype(np.float64) - mu) / sig
        w = 1.0 / (1.0 + np.abs(z))
        w = w * (len(w) / np.sum(w))
        return w.astype(np.float64)
    raise ValueError("sample_weight_mode must be 'none' or 'inv_var_tg', got %r" % mode)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_downstream_metrics_json(metrics_path, finetune_config, payload):
    """Save downstream metrics as JSON."""
    p = Path(metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": finetune_config.get("experiment_name", ""),
        "train_file": finetune_config.get("train_file"),
        "test_file": finetune_config.get("test_file"),
        "model_path": finetune_config.get("model_path"),
        "best_model_path": finetune_config.get("best_model_path"),
        "use_descriptor_fusion": finetune_config.get("use_descriptor_fusion"),
        "descriptor_ablation": finetune_config.get("descriptor_ablation"),
        "k": finetune_config.get("k"),
        "num_epochs": finetune_config.get("num_epochs"),
    }
    for opt_key in (
        "loss_type",
        "huber_beta",
        "quantile_tau",
        "seed",
        "LLRD_flag",
        "val_selection_metric",
        "cv_mode",
        "stratify_n_bins",
        "reg_head_hidden_mult",
        "sample_weight_mode",
        "data_version",
        "weight_decay",
    ):
        if opt_key in finetune_config and finetune_config[opt_key] is not None:
            record[opt_key] = finetune_config.get(opt_key)
    record.update(payload)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print("Saved metrics JSON ->", p.resolve())

"""Layer-wise learning rate decay"""

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)

"""Model"""

class DownstreamRegression(nn.Module):
    def __init__(
        self,
        pretrained_model,
        tokenizer,
        drop_rate=0.1,
        use_descriptor_fusion=False,
        n_desc=0,
        descriptor_ablation="none",
        descriptor_ablation_column=0,
        reg_head_hidden_mult=1.0,
    ):
        super(DownstreamRegression, self).__init__()
        self.use_descriptor_fusion = bool(use_descriptor_fusion) and int(n_desc) > 0
        self.n_desc = int(n_desc) if self.use_descriptor_fusion else 0
        self.descriptor_ablation = descriptor_ablation or "none"
        self.descriptor_ablation_column = int(descriptor_ablation_column)

        self.PretrainedModel = deepcopy(pretrained_model)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        h = self.PretrainedModel.config.hidden_size
        h_in = h + self.n_desc if self.use_descriptor_fusion else h
        mult = float(reg_head_hidden_mult)
        if mult <= 0:
            raise ValueError("reg_head_hidden_mult must be > 0, got %s" % mult)
        h_hidden = max(1, int(round(h * mult)))
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(h_in, h_hidden),
            nn.SiLU(),
            nn.Linear(h_hidden, 1),
        )

    def apply_descriptor_ablation(self, desc):
        if desc is None:
            return None
        mode = self.descriptor_ablation
        if mode == "none":
            return desc
        if mode == "zero_all":
            return desc * 0.0
        if mode == "drop_column":
            d = desc.clone()
            c = self.descriptor_ablation_column
            if 0 <= c < d.size(-1):
                d[:, c] = 0.0
            return d
        if mode == "shuffle_column":
            d = desc.clone()
            c = self.descriptor_ablation_column
            if 0 <= c < d.size(-1) and d.size(0) > 1:
                perm = torch.randperm(d.size(0), device=d.device)
                d[:, c] = d[perm, c]
            return d
        return desc

    def forward(self, input_ids, attention_mask, desc=None):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:, 0, :]
        if self.use_descriptor_fusion:
            if desc is None:
                raise ValueError("use_descriptor_fusion=True but desc is None")
            desc = self.apply_descriptor_ablation(desc)
            h = torch.cat([h, desc], dim=-1)
        return self.Regressor(h)

"""Train"""

def train(model, optimizer, scheduler, loss_fn, train_dataloader, device, finetune_config=None):

    model.train()
    use_desc = model.use_descriptor_fusion
    fc = finetune_config or {}

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prop = batch["prop"].to(device).float()
        desc = batch["desc"].to(device).float() if use_desc else None
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, desc).float()
        pred = outputs.view(-1).float()
        target = prop.view(-1).float()
        lt = (fc.get("loss_type") or "mse").lower().strip()
        if lt == "quantile":
            tau = float(fc.get("quantile_tau", 0.5))
            err = target - pred
            loss_per = torch.max(tau * err, (tau - 1) * err)
        elif lt == "huber":
            beta = float(fc.get("huber_beta", 1.0))
            loss_per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        else:
            loss_per = F.mse_loss(pred, target, reduction="none")
        if "weight" in batch:
            w = batch["weight"].to(device).float()
            loss = (loss_per * w).mean()
        else:
            loss = loss_per.mean() if loss_per.dim() > 0 else loss_per
        loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, loss_fn, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch):

    r2score = R2Score()
    train_loss = 0
    test_loss = 0
    mse_celsius = nn.MSELoss()
    # count = 0
    model.eval()
    with torch.no_grad():
        train_pred, train_true, test_pred, test_true = torch.tensor([]), torch.tensor([]), torch.tensor(
            []), torch.tensor([])

        use_desc = model.use_descriptor_fusion
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            desc = batch["desc"].to(device).float() if use_desc else None
            outputs = model(input_ids, attention_mask, desc).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = mse_celsius(outputs.squeeze(), prop.squeeze())
            train_loss += loss.item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), outputs.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().to("cpu"), train_true.flatten().to("cpu")).item()
        print("train RMSE = ", np.sqrt(train_loss))
        print("train r^2 = ", r2_train)

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            desc = batch["desc"].to(device).float() if use_desc else None
            outputs = model(input_ids, attention_mask, desc).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = mse_celsius(outputs.squeeze(), prop.squeeze())
            test_loss += loss.item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), outputs.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])

        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().to("cpu"), test_true.flatten().to("cpu")).item()
        print("test RMSE = ", np.sqrt(test_loss))
        print("test r^2 = ", r2_test)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("r^2/train", r2_train, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("r^2/test", r2_test, epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, finetune_config['save_path'])

    return train_loss, test_loss, r2_train, r2_test

    """

    if r2_test > best_test_r2:
        best_train_r2 = r2_train
        best_test_r2 = r2_test
        train_loss_best = train_loss
        test_loss_best = test_loss
        count = 0
    else:
        count += 1

    if r2_test > best_r2:
        best_r2 = r2_test
        torch.save(state, finetune_config['best_model_path'])         # save the best model

    if count >= finetune_config['tolerance']:
        print("Early stop")
        if best_test_r2 == 0:
            print("Poor performance with negative r^2")
            return None
        else:
            return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2

    return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2
    """

def main(finetune_config, pretrained_model, tokenizer):

    ensure_checkpoint_parent_dirs(finetune_config)

    """Tokenizer"""
    if finetune_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(finetune_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    best_r2 = 0.0           # monitor the best r^2 in the run (val_selection_metric=r2)
    best_val_mse_global = float("inf")  # monitor best val MSE when val_selection_metric=rmse

    """Data"""
    if finetune_config['CV_flag']:
        print("Start Cross Validation")
        data = pd.read_csv(finetune_config['train_file'])
        desc_cols = descriptor_cols_from_config(finetune_config)
        if finetune_config.get("use_descriptor_fusion"):
            assert_descriptor_columns_present(data, desc_cols)
        use_fusion = bool(finetune_config.get("use_descriptor_fusion")) and desc_cols
        n_desc = len(desc_cols) if use_fusion else 0

        cv_mode = (finetune_config.get("cv_mode") or "kfold").lower().strip()
        print("cv_mode =", cv_mode, "val_selection_metric =", val_selection_metric_from_config(finetune_config))
        split_list = build_cv_split_indices(finetune_config, data)
        train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = [], [], [], []     # monitor the best metrics in each fold
        for fold, (train_idx, val_idx) in enumerate(split_list):
            print('Fold {}'.format(fold + 1))
            fold_seed = int(finetune_config.get("seed", 42)) + fold
            set_training_seed(fold_seed)

            train_data = data.loc[train_idx, :].reset_index(drop=True)
            test_data = data.loc[val_idx, :].reset_index(drop=True)

            if finetune_config['aug_flag']:
                print("Data Augmentation")
                DataAug = DataAugmentation(finetune_config['aug_indicator'])
                train_data = DataAug.smiles_augmentation(train_data)
                if finetune_config['aug_special_flag']:
                    train_data = DataAug.smiles_augmentation_2(train_data)
                    train_data = DataAug.combine_smiles(train_data)
                    test_data = DataAug.combine_smiles(test_data)
                train_data = DataAug.combine_columns(train_data)
                test_data = DataAug.combine_columns(test_data)

            if use_fusion:
                train_data, test_data = scale_descriptor_block(train_data, test_data, desc_cols)

            scaler = StandardScaler()
            raw_train_y = train_data.iloc[:, 1].values.astype(np.float64)
            train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
            test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))

            train_sw = compute_train_sample_weights(
                raw_train_y, (finetune_config.get("sample_weight_mode") or "none")
            )
            train_dataset = Downstream_Dataset(
                train_data,
                tokenizer,
                finetune_config['blocksize'],
                descriptor_cols=desc_cols if use_fusion else None,
                sample_weights=train_sw,
            )
            test_dataset = Downstream_Dataset(
                test_data, tokenizer, finetune_config['blocksize'], descriptor_cols=desc_cols if use_fusion else None
            )
            _dl = dataloader_kw(finetune_config)
            train_dataloader = DataLoader(
                train_dataset, finetune_config["batch_size"], shuffle=True, **_dl
            )
            test_dataloader = DataLoader(
                test_dataset, finetune_config["batch_size"], shuffle=False, **_dl
            )

            """Parameters for scheduler"""
            steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
            training_steps = steps_per_epoch * finetune_config['num_epochs']
            warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

            """Train the model"""
            model = DownstreamRegression(
                pretrained_model,
                tokenizer,
                drop_rate=finetune_config['drop_rate'],
                use_descriptor_fusion=use_fusion,
                n_desc=n_desc,
                descriptor_ablation=finetune_config.get('descriptor_ablation', 'none'),
                descriptor_ablation_column=int(finetune_config.get('descriptor_ablation_column', 0)),
                reg_head_hidden_mult=float(finetune_config.get("reg_head_hidden_mult", 1.0)),
            ).to(device)
            model = model.float()
            loss_fn = build_training_loss(finetune_config)

            if finetune_config['LLRD_flag']:
                optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
            else:
                optimizer = AdamW(
                    [
                        {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                         "weight_decay": 0.0},
                        {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                         "weight_decay": finetune_config['weight_decay']},
                    ]
                )

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
            torch.cuda.empty_cache()
            train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0
            best_mse_fold = float("inf")
            fold_has_best = False
            count = 0     # Keep track of how many successive non-improvement epochs
            for epoch in range(finetune_config['num_epochs']):
                print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))
                train(model, optimizer, scheduler, loss_fn, train_dataloader, device, finetune_config)
                train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch)
                if validation_epoch_is_improvement(
                    finetune_config, r2_test, test_loss, best_test_r2, best_mse_fold, fold_has_best
                ):
                    best_train_r2 = r2_train
                    best_test_r2 = r2_test
                    train_loss_best = train_loss
                    test_loss_best = test_loss
                    best_mse_fold = float(test_loss)
                    fold_has_best = True
                    count = 0
                    state_fold = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "fold": fold,
                    }
                    fold_ckpt = Path(finetune_config["best_model_path"]).expanduser().parent / ("fold_%d_best.pt" % (fold + 1))
                    fold_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state_fold, fold_ckpt)
                else:
                    count += 1

                sel = val_selection_metric_from_config(finetune_config)
                if sel == "rmse":
                    if float(test_loss) < best_val_mse_global:
                        best_val_mse_global = float(test_loss)
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold': fold}
                        torch.save(state, finetune_config['best_model_path'])
                else:
                    if r2_test > best_r2:
                        best_r2 = r2_test
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold': fold}
                        torch.save(state, finetune_config['best_model_path'])

                if count >= finetune_config['tolerance']:
                    print("Early stop")
                    if val_selection_metric_from_config(finetune_config) == "r2" and best_test_r2 == 0:
                        print("Poor performance with negative r^2")
                    break

            train_loss_avg.append(np.sqrt(train_loss_best))
            test_loss_avg.append(np.sqrt(test_loss_best))
            train_r2_avg.append(best_train_r2)
            test_r2_avg.append(best_test_r2)
            writer.flush()

        """Average of metrics over all folds"""
        train_rmse = np.mean(np.array(train_loss_avg))
        test_rmse = np.mean(np.array(test_loss_avg))
        train_r2 = np.mean(np.array(train_r2_avg))
        test_r2 = np.mean(np.array(test_r2_avg))
        std_test_rmse = np.std(np.array(test_loss_avg))
        std_test_r2 = np.std(np.array(test_r2_avg))

        print("Train RMSE =", train_rmse)
        print("Test RMSE =", test_rmse)
        print("Train R^2 =", train_r2)
        print("Test R^2 =", test_r2)
        print("Standard Deviation of Test RMSE =", std_test_rmse)
        print("Standard Deviation of Test R^2 =", std_test_r2)

        mp = finetune_config.get("metrics_json_path")
        if mp:
            _sel = val_selection_metric_from_config(finetune_config)
            _note = (
                "val_* metrics are taken from each fold's best epoch according to "
                "val_selection_metric=%s; RMSE values are in deg C; cv_mode=%s"
                % (_sel, (finetune_config.get("cv_mode") or "kfold"))
            )
            save_downstream_metrics_json(
                mp,
                finetune_config,
                {
                    "cv_flag": True,
                    "per_fold_train_rmse": [float(x) for x in train_loss_avg],
                    "per_fold_val_rmse": [float(x) for x in test_loss_avg],
                    "per_fold_train_r2": [float(x) for x in train_r2_avg],
                    "per_fold_val_r2": [float(x) for x in test_r2_avg],
                    "mean_train_rmse": float(train_rmse),
                    "mean_val_rmse": float(test_rmse),
                    "mean_train_r2": float(train_r2),
                    "mean_val_r2": float(test_r2),
                    "std_val_rmse": float(std_test_rmse),
                    "std_val_r2": float(std_test_r2),
                    "note": _note,
                },
            )

    else:
        print("Train Test Split")
        train_data = pd.read_csv(finetune_config['train_file'])
        test_data = pd.read_csv(finetune_config['test_file'])
        desc_cols = descriptor_cols_from_config(finetune_config)
        if finetune_config.get("use_descriptor_fusion"):
            assert_descriptor_columns_present(train_data, desc_cols)
            assert_descriptor_columns_present(test_data, desc_cols)
        use_fusion = bool(finetune_config.get("use_descriptor_fusion")) and desc_cols
        n_desc = len(desc_cols) if use_fusion else 0

        if finetune_config['aug_flag']:
            print("Data Augmentation")
            DataAug = DataAugmentation(finetune_config['aug_indicator'])
            train_data = DataAug.smiles_augmentation(train_data)
            if finetune_config['aug_special_flag']:
                train_data = DataAug.smiles_augmentation_2(train_data)
                train_data = DataAug.combine_smiles(train_data)
                test_data = DataAug.combine_smiles(test_data)
            train_data = DataAug.combine_columns(train_data)
            test_data = DataAug.combine_columns(test_data)

        if use_fusion:
            train_data, test_data = scale_descriptor_block(train_data, test_data, desc_cols)

        scaler = StandardScaler()
        raw_train_y = train_data.iloc[:, 1].values.astype(np.float64)
        train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
        test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))

        train_sw = compute_train_sample_weights(
            raw_train_y, (finetune_config.get("sample_weight_mode") or "none")
        )
        train_dataset = Downstream_Dataset(
            train_data,
            tokenizer,
            finetune_config['blocksize'],
            descriptor_cols=desc_cols if use_fusion else None,
            sample_weights=train_sw,
        )
        test_dataset = Downstream_Dataset(
            test_data, tokenizer, finetune_config['blocksize'], descriptor_cols=desc_cols if use_fusion else None
        )
        _dl = dataloader_kw(finetune_config)
        train_dataloader = DataLoader(
            train_dataset, finetune_config["batch_size"], shuffle=True, **_dl
        )
        test_dataloader = DataLoader(
            test_dataset, finetune_config["batch_size"], shuffle=False, **_dl
        )

        """Parameters for scheduler"""
        steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
        training_steps = steps_per_epoch * finetune_config['num_epochs']
        warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

        """Train the model"""
        print("val_selection_metric =", val_selection_metric_from_config(finetune_config))
        model = DownstreamRegression(
            pretrained_model,
            tokenizer,
            drop_rate=finetune_config['drop_rate'],
            use_descriptor_fusion=use_fusion,
            n_desc=n_desc,
            descriptor_ablation=finetune_config.get('descriptor_ablation', 'none'),
            descriptor_ablation_column=int(finetune_config.get('descriptor_ablation_column', 0)),
            reg_head_hidden_mult=float(finetune_config.get("reg_head_hidden_mult", 1.0)),
        ).to(device)
        model = model.float()
        loss_fn = build_training_loss(finetune_config)
        set_training_seed(int(finetune_config.get("seed", 42)))

        if finetune_config['LLRD_flag']:
            optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                     "weight_decay": 0.0},
                    {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                     "weight_decay": finetune_config['weight_decay']},
                ]
            )

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        torch.cuda.empty_cache()
        train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0
        best_mse_fold = float("inf")
        split_has_best = False
        count = 0     # Keep track of how many successive non-improvement epochs
        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1,finetune_config['num_epochs']))
            train(model, optimizer, scheduler, loss_fn, train_dataloader, device, finetune_config)
            train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch)
            if validation_epoch_is_improvement(
                finetune_config, r2_test, test_loss, best_test_r2, best_mse_fold, split_has_best
            ):
                best_train_r2 = r2_train
                best_test_r2 = r2_test
                train_loss_best = train_loss
                test_loss_best = test_loss
                best_mse_fold = float(test_loss)
                split_has_best = True
                count = 0
            else:
                count += 1

            sel = val_selection_metric_from_config(finetune_config)
            if sel == "rmse":
                if float(test_loss) < best_val_mse_global:
                    best_val_mse_global = float(test_loss)
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                    torch.save(state, finetune_config['best_model_path'])
            else:
                if r2_test > best_r2:
                    best_r2 = r2_test
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                    torch.save(state, finetune_config['best_model_path'])

            if count >= finetune_config['tolerance']:
                print("Early stop")
                if val_selection_metric_from_config(finetune_config) == "r2" and best_test_r2 == 0:
                    print("Poor performance with negative r^2")
                break

        writer.flush()

        mp = finetune_config.get("metrics_json_path")
        if mp:
            save_downstream_metrics_json(
                mp,
                finetune_config,
                {
                    "cv_flag": False,
                    "best_train_rmse": float(np.sqrt(train_loss_best)) if train_loss_best > 0 else None,
                    "best_val_rmse": float(np.sqrt(test_loss_best)) if test_loss_best > 0 else None,
                    "best_train_r2": float(best_train_r2),
                    "best_val_r2": float(best_test_r2),
                    "note": "best_* early stopvalidationbest; unit RMSE label ( deg C)",
                },
            )


if __name__ == "__main__":

    _ap = argparse.ArgumentParser(description="TransPolymer downstream finetune")
    _ap.add_argument(
        "--config",
        type=str,
        default="config_finetune.yaml",
        help=" YAML path (currentdirectory, directory)",
    )
    _args = _ap.parse_args()
    with open(_args.config, "r", encoding="utf-8") as _f:
        finetune_config = yaml.load(_f, Loader=yaml.FullLoader)
    print("config file:", _args.config)
    print(finetune_config)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    if finetune_config['model_indicator'] == 'pretrain':
        print("Use the pretrained model")
        model_path = str(finetune_config['model_path'])
        PretrainedModel = RobertaModel.from_pretrained(
            model_path,
            local_files_only=Path(model_path).exists(),
        )
        tok_path = Path("./roberta-base")
        if not tok_path.is_dir():
            tok_path = Path(model_path)
        tokenizer = PolymerSmilesTokenizer.from_pretrained(
            str(tok_path),
            max_len=finetune_config['blocksize'],
            local_files_only=tok_path.exists(),
        )
        PretrainedModel.config.hidden_dropout_prob = finetune_config['hidden_dropout_prob']
        PretrainedModel.config.attention_probs_dropout_prob = finetune_config['attention_probs_dropout_prob']
    else:
        print("No Pretrain")
        config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        PretrainedModel = RobertaModel(config=config)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
    max_token_len = finetune_config['blocksize']

    """Run the main function"""
    main(finetune_config, PretrainedModel, tokenizer)





