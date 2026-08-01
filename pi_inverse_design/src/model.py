

import math
import torch
import torch.nn as nn


class PIGeneratorModel(nn.Module):
    def __init__(
        self,
        vocab_size:      int,
        d_model:         int   = 256,
        nhead:           int   = 8,
        num_enc_layers:  int   = 2,
        num_dec_layers:  int   = 4,
        d_ff:            int   = 1024,
        dropout:         float = 0.1,
        max_len:         int   = 300,
        num_bins:        int   = 20,
        pad_idx:         int   = 0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.pad_idx   = pad_idx
        self.vocab_size = vocab_size

        # ── 1. Tg items ────────────────────────────────────────
        # 1a.  MLP:   (B,1) → (B, d_model)
        self.tg_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 1b. Bin Embedding:   (B,) → (B, d_model)
        # num_bins+1  digitize 
        self.tg_bin_emb = nn.Embedding(num_bins + 2, d_model)
        # 1c. 
        self.cond_norm = nn.LayerNorm(d_model)

        # ── 2. Token Embedding +  ──────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # ── 3. Transformer Encoder ( Tg items)────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_ff,
            dropout        = dropout,
            batch_first    = True,
            activation     = "gelu",
            norm_first     = True,    # Pre-LN, training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ── 4. Transformer Decoder (generate)─────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_ff,
            dropout        = dropout,
            batch_first    = True,
            activation     = "gelu",
            norm_first     = True,    # Pre-LN
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # ── 5. output ────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        # checkpoint: output token_emb checkpoint (, model)
        self.out_proj.weight = self.token_emb.weight

        # 
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for name, p in self.named_parameters():
            if "out_proj" in name:
                continue          # checkpoint, 
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Embedding 
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.01)
        nn.init.normal_(self.tg_bin_emb.weight,mean=0.0, std=0.01)

    # ─────────────────────────────────────────────────────────────────
    def encode_tg(self, tg_norm: torch.Tensor, tg_bin_id: torch.Tensor):
        """
         Tg  memory,  Decoder .

        Args:
            tg_norm   : (B,)    Tg 
            tg_bin_id : (B,)   binning ID
        Returns:
            memory    : (B, 1, d_model)
        """
        # path: (B,) → (B, 1) → MLP → (B, d_model)
        scalar_emb = self.tg_mlp(tg_norm.unsqueeze(-1))          # (B, d_model)
        # binningpath: (B,) → Embedding → (B, d_model)
        bin_emb    = self.tg_bin_emb(tg_bin_id)                  # (B, d_model)
        #  + LayerNorm
        fused      = self.cond_norm(scalar_emb + bin_emb)        # (B, d_model)
        # column: (B, 1, d_model)
        memory_raw = fused.unsqueeze(1)
        #  Encoder 
        memory     = self.encoder(memory_raw)                     # (B, 1, d_model)
        return memory

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        tg_norm:             torch.Tensor,   # (B,)
        tg_bin_id:           torch.Tensor,   # (B,)
        tgt_tokens:          torch.Tensor,   # (B, T)
        tgt_key_padding_mask: torch.Tensor = None,  # (B, T) True=PAD
    ):
        """
        trainingforward.
        Returns:
            logits : (B, T, vocab_size)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        # ── Tg items → Memory ──────────────────────────────────
        memory = self.encode_tg(tg_norm, tg_bin_id)    # (B, 1, d_model)

        # ── Token Embedding +  ────────────────────────────
        positions = torch.arange(T, device=device)
        x = self.emb_drop(
            self.token_emb(tgt_tokens) + self.pos_emb(positions)
        )                                               # (B, T, d_model)

        # ──  Mask (, )─────────────────────
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device
        )                                               # (T, T)

        # ── Decoder ──────────────────────────────────────────────
        out = self.decoder(
            tgt                  = x,
            memory               = memory,
            tgt_mask             = causal_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
        )                                               # (B, T, d_model)

        # ── output ─────────────────────────────────────────────────
        out    = self.out_norm(out)
        logits = self.out_proj(out)                     # (B, T, vocab_size)
        return logits

    # ─────────────────────────────────────────────────────────────────
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"training: {total / 1e6:.2f} M")
        return total


def build_model(vocab_size: int, stats: dict, device: str = "cpu", **kwargs) -> PIGeneratorModel:
    """:  stats  n_bins, model"""
    n_bins = stats.get("n_bins", 20)
    model  = PIGeneratorModel(vocab_size=vocab_size, num_bins=n_bins, **kwargs)
    model  = model.to(device)
    model.count_parameters()
    return model
