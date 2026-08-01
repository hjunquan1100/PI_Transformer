"""
train.py

Train the target-Tg-conditioned PI generator.

  - AdamW + Warmup + CosineAnnealingLR
  - Label Smoothing CrossEntropyLoss
  - Early Stopping (patience)
  - Best-checkpoint saving
  - TensorBoard log

Example:
    python src/train.py --d_model 256 --num_dec_layers 4 --epochs 200 --batch_size 32 --patience 50
"""

import os
import json
import math
import argparse
import time
import csv
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import build_dataloaders
from model   import build_model
from paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_PROCESSED_DIR


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for tgt_in, tgt_out, pad_mask, tg_norm, tg_bin in loader:
        tgt_in   = tgt_in.to(device)
        tgt_out  = tgt_out.to(device)
        pad_mask = pad_mask.to(device)
        tg_norm  = tg_norm.to(device)
        tg_bin   = tg_bin.to(device)

        optimizer.zero_grad()

        logits = model(tg_norm, tg_bin, tgt_in, tgt_key_padding_mask=pad_mask)
        # logits: (B, T, vocab_size)  tgt_out: (B, T)

        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # Ignore PAD tokens when averaging token loss.
        n_tokens = (tgt_out != 0).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for tgt_in, tgt_out, pad_mask, tg_norm, tg_bin in loader:
        tgt_in   = tgt_in.to(device)
        tgt_out  = tgt_out.to(device)
        pad_mask = pad_mask.to(device)
        tg_norm  = tg_norm.to(device)
        tg_bin   = tg_bin.to(device)

        logits = model(tg_norm, tg_bin, tgt_in, tgt_key_padding_mask=pad_mask)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        n_tokens = (tgt_out != 0).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def save_history(history, json_path, csv_path):
    """Save training history as JSON and CSV."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def best_model_path(ckpt_dir, val_loss):
    """Build the best-checkpoint path from validation loss."""
    return os.path.join(ckpt_dir, f"best_model_valloss_{val_loss:.4f}.pt")


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(args.ckpt_dir, f"reverse_{run_timestamp}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    print(f"Training output directory: {run_ckpt_dir}")

    # Data.
    train_loader, val_loader, vocab, inv_vocab, stats = build_dataloaders(
        processed_dir = args.processed_dir,
        batch_size    = args.batch_size,
        max_len       = args.max_len,
        num_workers   = args.num_workers,
    )
    vocab_size = len(vocab)

    # Model.
    model = build_model(
        vocab_size      = vocab_size,
        stats           = stats,
        device          = str(device),
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_enc_layers  = args.num_enc_layers,
        num_dec_layers  = args.num_dec_layers,
        d_ff            = args.d_ff,
        dropout         = args.dropout,
        max_len         = args.max_len,
        pad_idx         = vocab["<PAD>"],
    )

    # Loss.
    criterion = nn.CrossEntropyLoss(
        ignore_index   = vocab["<PAD>"],
        label_smoothing= args.label_smoothing,
    )

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        betas        = (0.9, 0.98),
        weight_decay = args.weight_decay,
    )

    # Learning-rate schedule.
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}  warmup steps: {warmup_steps}")

    # TensorBoard and training history.
    writer = SummaryWriter(log_dir=os.path.join(run_ckpt_dir, "tb_logs"))
    hist_path = os.path.join(run_ckpt_dir, "train_history.json")
    hist_csv_path = os.path.join(run_ckpt_dir, "train_history.csv")

    # Training loop.
    best_val_loss = float("inf")
    best_epoch = 0
    current_best_path = None
    no_improve    = 0
    history       = []

    print("\ntraining...\n" + "=" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, grad_clip=args.grad_clip,
        )
        val_loss, val_ppl = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        is_best = val_loss < best_val_loss
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_ppl":    round(val_ppl, 2),
            "lr":         round(current_lr, 8),
            "elapsed_sec": round(elapsed, 2),
            "is_best":    is_best,
        })
        save_history(history, hist_path, hist_csv_path)

        writer.add_scalar("Loss/train",      train_loss, epoch)
        writer.add_scalar("Loss/val",        val_loss,   epoch)
        writer.add_scalar("Perplexity/val",  val_ppl,    epoch)
        writer.add_scalar("LR",              current_lr, epoch)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ppl={val_ppl:.2f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if is_best:
            prev_best_path = current_best_path
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve    = 0
            current_best_path = best_model_path(run_ckpt_dir, best_val_loss)
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "train_loss":      train_loss,
                "vocab":           vocab,
                "inv_vocab":       inv_vocab,
                "stats":           stats,
                "model_config": {
                    "vocab_size":     vocab_size,
                    "d_model":        args.d_model,
                    "nhead":          args.nhead,
                    "num_enc_layers": args.num_enc_layers,
                    "num_dec_layers": args.num_dec_layers,
                    "d_ff":           args.d_ff,
                    "dropout":        args.dropout,
                    "max_len":        args.max_len,
                    "num_bins":       stats.get("n_bins", 20),
                    "pad_idx":        vocab["<PAD>"],
                },
            }, current_best_path)
            if prev_best_path and prev_best_path != current_best_path and os.path.exists(prev_best_path):
                os.remove(prev_best_path)
            print(f"  Saved best model: {os.path.basename(current_best_path)}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: no validation-loss improvement for {args.patience} epochs.")
                break

    writer.close()
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print("Primary model-selection metric: validation loss")
    print(f"Checkpoint saved: {current_best_path}")
    print(f"Training directory: {run_ckpt_dir}")
    print(f"Training history JSON: {hist_path}")
    print(f"Training history CSV: {hist_csv_path}")
    return model, vocab, inv_vocab, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PI inverse-design generator")

    # data
    parser.add_argument("--processed_dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--ckpt_dir",      default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--max_len",       type=int, default=300)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=0)

    # model
    parser.add_argument("--d_model",        type=int,   default=256)
    parser.add_argument("--nhead",          type=int,   default=8)
    parser.add_argument("--num_enc_layers", type=int,   default=2)
    parser.add_argument("--num_dec_layers", type=int,   default=4)
    parser.add_argument("--d_ff",           type=int,   default=1024)
    parser.add_argument("--dropout",        type=float, default=0.1)

    # training
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--warmup_ratio",    type=float, default=0.1,
                        help="Fraction of total steps used for warmup")
    parser.add_argument("--patience",        type=int,   default=20,
                        help="Early-stopping patience in epochs")

    args = parser.parse_args()
    train(args)
