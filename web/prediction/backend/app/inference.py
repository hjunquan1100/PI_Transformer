"""TransPolymer PI Tg single-SMILES inference (singleton)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from transformers import RobertaModel

from app.config import ARTIFACTS_DIR, PI_FORWARD_ROOT, PI_TG_CKPT, PI_TG_CONFIG, SCALERS_PATH
from app.descriptors import SmilesDescriptorError, row_for_descriptor_cols

if str(PI_FORWARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PI_FORWARD_ROOT))

from Downstream import (  # noqa: E402
    DownstreamRegression,
    descriptor_cols_from_config,
)
from PolymerSmilesTokenization import PolymerSmilesTokenizer  # noqa: E402


class TgPredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(PI_TG_CONFIG, "r", encoding="utf-8") as f:
            self.cfg: dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

        self.desc_cols = descriptor_cols_from_config(self.cfg) or []
        self.use_fusion = bool(self.cfg.get("use_descriptor_fusion")) and bool(self.desc_cols)
        mfp_cols = [c for c in self.desc_cols if c.startswith("mfp_")]
        if mfp_cols:
            self.morgan_bits = max(int(c.split("_")[1]) for c in mfp_cols) + 1
        else:
            self.morgan_bits = 128

        self._load_model()
        self._load_scalers()

    def _load_model(self) -> None:
        model_path = PI_FORWARD_ROOT / str(self.cfg["model_path"])
        pretrained = RobertaModel.from_pretrained(str(model_path), local_files_only=True)
        pretrained.config.hidden_dropout_prob = self.cfg["hidden_dropout_prob"]
        pretrained.config.attention_probs_dropout_prob = self.cfg["attention_probs_dropout_prob"]

        ckpt_path = PI_TG_CKPT
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location=self.device)

        # Tokenizer files ship with the contrastive backbone (or ./roberta-base if present).
        tok_path = model_path
        roberta_local = PI_FORWARD_ROOT / "roberta-base"
        if roberta_local.is_dir():
            tok_path = roberta_local
        self.tokenizer = PolymerSmilesTokenizer.from_pretrained(
            str(tok_path),
            max_len=int(self.cfg["blocksize"]),
            local_files_only=True,
        )
        if self.cfg.get("add_vocab_flag"):
            vocab_sup = (
                pd.read_csv(PI_FORWARD_ROOT / str(self.cfg["vocab_sup_file"]), header=None)
                .values.flatten()
                .tolist()
            )
            self.tokenizer.add_tokens(vocab_sup)
        self._align_tokenizer_to_checkpoint(ckpt)

        n_desc = len(self.desc_cols) if self.use_fusion else 0
        self.model = DownstreamRegression(
            pretrained,
            self.tokenizer,
            drop_rate=float(self.cfg["drop_rate"]),
            use_descriptor_fusion=self.use_fusion,
            n_desc=n_desc,
            descriptor_ablation=self.cfg.get("descriptor_ablation", "none"),
            descriptor_ablation_column=int(self.cfg.get("descriptor_ablation_column", 0)),
            reg_head_hidden_mult=float(self.cfg.get("reg_head_hidden_mult", 1.0)),
        ).to(self.device)
        self.model = self.model.float()

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _align_tokenizer_to_checkpoint(self, ckpt: dict[str, Any]) -> None:
        """Pad tokenizer with inert tokens when the checkpoint used a larger vocab."""
        target_vocab_size = None
        for key, value in ckpt.get("model", {}).items():
            if key.endswith("embeddings.word_embeddings.weight"):
                target_vocab_size = int(value.shape[0])
                break
        if target_vocab_size is None:
            return

        current_vocab_size = len(self.tokenizer)
        if current_vocab_size > target_vocab_size:
            raise ValueError(
                "Tokenizer vocabulary is larger than checkpoint embeddings: "
                f"{current_vocab_size} > {target_vocab_size}"
            )
        if current_vocab_size == target_vocab_size:
            return

        missing = target_vocab_size - current_vocab_size
        extra_tokens = [f"<ckpt_extra_{i}>" for i in range(missing)]
        added = self.tokenizer.add_tokens(extra_tokens)
        if len(self.tokenizer) != target_vocab_size or added != missing:
            raise ValueError(
                "Could not align tokenizer vocabulary to checkpoint embeddings: "
                f"added={added}, current={len(self.tokenizer)}, target={target_vocab_size}"
            )

    def _load_scalers(self) -> None:
        if SCALERS_PATH.is_file():
            data = joblib.load(SCALERS_PATH)
            self.desc_scaler: StandardScaler | None = data.get("desc_scaler")
            self.y_scaler: StandardScaler = data["y_scaler"]
            self.desc_cols = data.get("desc_cols", self.desc_cols)
            return

        train_csv = PI_FORWARD_ROOT / str(self.cfg["train_file"])
        if not train_csv.is_file():
            raise FileNotFoundError(f"Training CSV for scalers: {train_csv}")

        df = pd.read_csv(train_csv)
        for c in self.desc_cols:
            if c not in df.columns:
                raise ValueError(f"Column {c!r} missing in {train_csv}")

        self.desc_scaler = StandardScaler()
        self.desc_scaler.fit(df[self.desc_cols].values)

        self.y_scaler = StandardScaler()
        self.y_scaler.fit(df.iloc[:, 1].values.reshape(-1, 1))

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "desc_scaler": self.desc_scaler,
                "y_scaler": self.y_scaler,
                "desc_cols": self.desc_cols,
            },
            SCALERS_PATH,
        )

    def predict_tg(self, smiles: str) -> float:
        from app.descriptors import canonicalize_smiles

        smiles = canonicalize_smiles(str(smiles).strip())
        if not smiles:
            raise SmilesDescriptorError("SMILES cannot be empty.")

        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=int(self.cfg["blocksize"]),
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        desc_tensor = None
        if self.use_fusion:
            row = row_for_descriptor_cols(smiles, self.desc_cols, morgan_bits=self.morgan_bits)
            desc_arr = row.values.reshape(1, -1)
            assert self.desc_scaler is not None
            desc_scaled = self.desc_scaler.transform(desc_arr)
            desc_tensor = torch.tensor(desc_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask, desc_tensor).float()
            out_np = out.cpu().numpy().reshape(-1, 1)
            tg = float(self.y_scaler.inverse_transform(out_np)[0, 0])

        return tg


@lru_cache(maxsize=1)
def get_predictor() -> TgPredictor:
    return TgPredictor()
