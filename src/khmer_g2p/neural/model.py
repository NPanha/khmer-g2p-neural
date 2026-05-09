"""Transformer encoder-decoder for Khmer grapheme-to-phoneme.

Defaults target a ~10k–50k-pair Khmer lexicon and aim for
val_PER < 0.05 / val_WER < 0.10:
    d_model=384, 4 encoder / 4 decoder layers, 8 heads, FFN 1536, dropout 0.3.
An optional CTC auxiliary head (`use_ctc=True`) is projected off the
encoder output. During training the main loss is
    loss = CE(decoder) + ctc_weight * CTC(encoder)
which regularizes the encoder toward a monotonic alignment and is typically
worth 1–2 PER points on low-resource G2P.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class G2PConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 384
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.3
    max_len: int = 256
    pad_id: int = 0
    # CTC auxiliary head on the encoder. Off by default so old
    # checkpoints (pre-0.3) still load; TrainConfig turns it on.
    use_ctc: bool = False
    # Tie the decoder input embedding and the output projection weights.
    # Saves params and usually improves PER. Off by default so old
    # (pre-0.4) checkpoints still load; TrainConfig turns it on for v0.4+.
    tie_embeddings: bool = False
    # 0.5: encoder-side auxiliary heads. Off by default so old checkpoints
    # still load. TrainConfig turns each on with its own loss weight.
    use_series_head: bool = False         # 2-class: series 1 vs series 2
    use_syllable_head: bool = False       # 2-class: syllable-boundary vs not

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "G2PConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class PositionalEmbedding(nn.Module):
    """Learned positional embedding, capped at max_len."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class G2PTransformer(nn.Module):
    """Character-level encoder-decoder Transformer for grapheme→phoneme."""

    def __init__(self, cfg: G2PConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.src_pos = PositionalEmbedding(cfg.max_len, cfg.d_model)
        self.tgt_pos = PositionalEmbedding(cfg.max_len, cfg.d_model)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

        # Optional CTC auxiliary head on the encoder output.
        # Blank = pad_id, which never appears in real target sequences.
        self.ctc_proj: Optional[nn.Linear] = None
        if cfg.use_ctc:
            self.ctc_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

        # 0.5: encoder-side structural prediction heads. Each is a tiny
        # per-position classifier: encoder memory → 2 classes.
        # See khmer_g2p.neural.aux_labels for the label scheme.
        self.series_head: Optional[nn.Linear] = None
        if cfg.use_series_head:
            self.series_head = nn.Linear(cfg.d_model, 2)
        self.syllable_head: Optional[nn.Linear] = None
        if cfg.use_syllable_head:
            self.syllable_head = nn.Linear(cfg.d_model, 2)

        self._init_weights()
        # Tying must happen *after* init so both pieces start from the same
        # (initialized) tensor. Re-call this method after load_state_dict to
        # restore tying after a checkpoint load.
        self.tie_weights_if_configured()

    def tie_weights_if_configured(self) -> None:
        """Tie decoder input embedding and output projection weights.

        Idempotent — calling it twice is a no-op once tied. Required after
        any ``load_state_dict`` because PyTorch copies into each parameter
        independently, which silently un-shares the storage.
        """
        if not self.cfg.tie_embeddings:
            return
        # Both should be (V_tgt, d_model). The Linear weight layout is
        # (out_features, in_features) which matches the embedding's (V, D).
        if self.tgt_embed.weight.shape != self.out_proj.weight.shape:
            raise RuntimeError(
                "tie_embeddings requires tgt_embed.weight and out_proj.weight "
                f"to have the same shape; got {tuple(self.tgt_embed.weight.shape)} "
                f"vs {tuple(self.out_proj.weight.shape)}"
            )
        self.out_proj.weight = self.tgt_embed.weight

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --- Masks -----------------------------------------------------------------

    def _key_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x == self.cfg.pad_id  # (B, T)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    # --- Forward ---------------------------------------------------------------

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        s = self.src_embed(src) * math.sqrt(self.cfg.d_model)
        s = self.src_pos(s)
        src_kpm = self._key_padding_mask(src)
        memory = self.transformer.encoder(s, src_key_padding_mask=src_kpm)
        return memory, src_kpm

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_kpm: torch.Tensor,
    ) -> torch.Tensor:
        t = self.tgt_embed(tgt_in) * math.sqrt(self.cfg.d_model)
        t = self.tgt_pos(t)
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)
        tgt_kpm = self._key_padding_mask(tgt_in)
        out = self.transformer.decoder(
            t,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.out_proj(out)  # (B, T, V)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_kpm = self.encode(src)
        return self.decode(tgt_in, memory, src_kpm)

    # --- CTC auxiliary head ----------------------------------------------------

    def series_logits(self, memory: torch.Tensor) -> torch.Tensor:
        """Per-position 2-class logits for the consonant-series head.

        Input  ``memory``  : (B, T_src, d_model)
        Output ``logits``  : (B, T_src, 2)
        """
        if self.series_head is None:
            raise RuntimeError(
                "series_head not enabled. Build the model with G2PConfig(use_series_head=True)."
            )
        return self.series_head(memory)

    def syllable_logits(self, memory: torch.Tensor) -> torch.Tensor:
        """Per-position 2-class logits for the syllable-boundary head."""
        if self.syllable_head is None:
            raise RuntimeError(
                "syllable_head not enabled. Build the model with G2PConfig(use_syllable_head=True)."
            )
        return self.syllable_head(memory)

    def ctc_log_probs(self, memory: torch.Tensor) -> torch.Tensor:
        """Project encoder memory to per-timestep log-probs for CTC loss.

        Input:   memory  (B, T_src, d_model)
        Output:  log_probs (T_src, B, V)  — the layout nn.functional.ctc_loss wants.
        """
        if self.ctc_proj is None:
            raise RuntimeError(
                "CTC head not enabled. Build the model with G2PConfig(use_ctc=True)."
            )
        logits = self.ctc_proj(memory)                  # (B, T_src, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1).contiguous()    # (T_src, B, V)

    # --- Parameter count (useful in logs) --------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
