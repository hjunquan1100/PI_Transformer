"""
dataset.py

PIDataset + collate_fn
"""

import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from paths import DEFAULT_PROCESSED_DIR

try:
    import selfies as sf
except ImportError:
    raise ImportError("Please install selfies: pip install selfies")


class PIDataset(Dataset):
    """Dataset of SELFIES strings paired with Tg values."""

    def __init__(
        self,
        data: list,          # list of (selfies_str, tg_value)
        vocab: dict,
        tg_mean: float,
        tg_std: float,
        tg_bins: list,       # n_bins + 1 bin edges
        max_len: int = 300,
    ):
        self.data    = data
        self.vocab   = vocab
        self.tg_mean = tg_mean
        self.tg_std  = tg_std
        self.tg_bins = np.array(tg_bins)
        self.max_len = max_len
        self.n_bins  = len(tg_bins) - 1

        self.PAD = vocab["<PAD>"]
        self.BOS = vocab["<BOS>"]
        self.EOS = vocab["<EOS>"]
        self.UNK = vocab["<UNK>"]

    def _encode(self, selfies_str: str) -> list:
        """Encode SELFIES to token ids with BOS/EOS and max-length truncation."""
        tokens = list(sf.split_selfies(selfies_str))
        ids = (
            [self.BOS]
            + [self.vocab.get(t, self.UNK) for t in tokens]
            + [self.EOS]
        )
        return ids[: self.max_len]

    def _tg_bin(self, tg: float) -> int:
        """Return Tg bin id in [0, n_bins - 1]."""
        idx = int(np.digitize(tg, self.tg_bins)) - 1
        return max(0, min(idx, self.n_bins - 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        selfies_str, tg = self.data[idx]
        ids     = self._encode(selfies_str)
        tg_norm = (tg - self.tg_mean) / self.tg_std
        tg_bin  = self._tg_bin(tg)
        return (
            torch.tensor(ids,     dtype=torch.long),
            torch.tensor(tg_norm, dtype=torch.float),
            torch.tensor(tg_bin,  dtype=torch.long),
        )


def collate_fn(batch):
    """Pad a batch and build decoder input/output tensors."""
    ids_list, tg_norm_list, tg_bin_list = zip(*batch)

    # PAD id is 0.
    padded   = pad_sequence(ids_list, batch_first=True, padding_value=0)
    tgt_in   = padded[:, :-1]
    tgt_out  = padded[:, 1:]
    pad_mask = tgt_in == 0

    return (
        tgt_in,
        tgt_out,
        pad_mask,
        torch.stack(tg_norm_list),
        torch.stack(tg_bin_list),
    )


def build_dataloaders(
    processed_dir: str = str(DEFAULT_PROCESSED_DIR),
    batch_size: int    = 64,
    max_len: int       = 300,
    num_workers: int   = 0,
):
    """Build train and validation DataLoaders from a processed dataset."""
    with open(f"{processed_dir}/vocab.json", encoding="utf-8") as f:
        vdata    = json.load(f)
        vocab    = vdata["vocab"]
        inv_vocab = {int(k): v for k, v in vdata["inv_vocab"].items()}

    with open(f"{processed_dir}/stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    with open(f"{processed_dir}/train_augmented.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open(f"{processed_dir}/val.pkl", "rb") as f:
        val_data = pickle.load(f)

    ds_kwargs = dict(
        vocab    = vocab,
        tg_mean  = stats["tg_mean"],
        tg_std   = stats["tg_std"],
        tg_bins  = stats["tg_bins"],
        max_len  = max_len,
    )

    train_ds = PIDataset(train_data, **ds_kwargs)
    val_ds   = PIDataset(val_data,   **ds_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    print(f"DataLoader ready  train={len(train_ds)}  val={len(val_ds)}  vocab={len(vocab)}")
    return train_loader, val_loader, vocab, inv_vocab, stats
