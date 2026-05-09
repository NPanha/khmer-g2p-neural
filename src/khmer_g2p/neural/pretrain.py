"""Masked-character pretraining for the Khmer encoder.

Self-supervised stage: take a monolingual Khmer corpus (e.g. a Wikipedia /
news dump), build a character-level vocabulary, then train an encoder-only
Transformer with a BERT-style masked-character objective::

    1. For each input position, with probability ``mlm_prob``, pick it for
       prediction.
    2. Of the picked positions: 80% are replaced by a [MASK] sentinel
       (we re-use the source ``UNK`` token as the sentinel — see note below),
       10% are replaced by a random character, 10% are left unchanged.
    3. Cross-entropy loss is computed only over the picked positions.

After pretraining, the encoder weights (plus the source embedding and the
positional embedding) are warm-started into the supervised G2P training.
For a 10–30k-pair Khmer lexicon, this typically buys 1–3 absolute PER on
top of the v0.4 recipe — the encoder enters supervised training already
"speaking Khmer" rather than starting from Xavier-uniform noise.

Why re-use UNK as the mask sentinel?
    Adding a new ``[MASK]`` token to ``Vocab.SPECIALS`` would change the
    serialization layout and break older checkpoints. ``UNK`` is otherwise
    rare in a clean Khmer character vocabulary built from a large corpus,
    so re-using it keeps the schema flat. If you'd rather have a dedicated
    sentinel, that's a small ``vocab.py`` change for a future release.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from khmer_g2p.lexicon import load_tsv
from khmer_g2p.neural.train import _cosine_lr, _resolve_amp
from khmer_g2p.neural.vocab import TOKENIZER_CHAR, Vocab
from khmer_g2p.normalizer import normalize


try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(it=None, *_, **__):
        return it if it is not None else iter([])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    corpus_path: str                       # plain-text file (one document per line)
    out_dir: str = "checkpoints_pretrain"
    lexicon_path: Optional[str] = None     # if given, vocab covers the lexicon too

    # Architecture (must match the supervised encoder for warm-start to work)
    d_model: int = 384
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_len: int = 256

    # MLM
    mlm_prob: float = 0.15
    window: int = 128                      # tokens per training example
    stride: int = 96                       # window stride (sliding overlap)
    min_chars_per_window: int = 8

    # Training
    batch_size: int = 64
    epochs: int = 10
    lr: float = 5e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Runtime
    device: str = "auto"                   # "auto" | "cpu" | "cuda"
    amp: str = "auto"                      # "auto" | "off" | "fp16" | "bf16"
    seed: int = 42
    num_workers: int = 0
    progress: bool = True


# ---------------------------------------------------------------------------
# Encoder-only model with an MLM head
# ---------------------------------------------------------------------------

class _EncoderOnly(nn.Module):
    """Mirrors G2PTransformer's encoder stack so weights transfer cleanly.

    Parameter names are chosen to match ``G2PTransformer`` 1-for-1 — the
    supervised loader (``train._load_pretrained_encoder``) just looks for
    ``src_embed.``, ``src_pos.``, and ``transformer.encoder.`` prefixes and
    copies them over.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        # Mirror G2PTransformer.PositionalEmbedding here (avoid a cross-import
        # cycle and keep this module standalone).
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.src_pos = _LearnedPos(max_len, d_model)
        # We only need an *encoder* Transformer for MLM. We construct a full
        # nn.Transformer with 1 dummy decoder layer so the parameter naming
        # matches G2PTransformer.transformer.encoder.* exactly.
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=1,            # unused, but keeps key names aligned
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)
        # Tie MLM head weights to the input embedding — standard MLM trick,
        # halves the head's params and tends to learn faster.
        self.mlm_head.weight = self.src_embed.weight
        self.pad_id = pad_id
        self.d_model = d_model
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        kpm = src == self.pad_id
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.src_pos(x)
        memory = self.transformer.encoder(x, src_key_padding_mask=kpm)
        return self.mlm_head(memory)        # (B, T, V)


class _LearnedPos(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _MaskedCharDataset(Dataset):
    """Slides a fixed-size window over a corpus of normalized character ids.

    Each ``__getitem__`` returns one window of token ids (no masking yet —
    masking is done by the collate function so each epoch sees fresh masks).
    """

    def __init__(self, corpus_ids: List[int], window: int, stride: int,
                 min_chars: int) -> None:
        if not corpus_ids:
            raise ValueError("empty corpus")
        self.window = window
        self.stride = max(1, stride)
        starts: List[int] = []
        n = len(corpus_ids)
        i = 0
        while i < n:
            j = min(i + window, n)
            if j - i >= min_chars:
                starts.append(i)
            if j >= n:
                break
            i += self.stride
        if not starts:
            starts = [0]                                          # tiny corpus
        self._corpus = corpus_ids
        self._starts = starts

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self._starts[idx]
        e = min(s + self.window, len(self._corpus))
        return torch.tensor(self._corpus[s:e], dtype=torch.long)


def _make_mask_collate(vocab: Vocab, mlm_prob: float, mask_id: int):
    """Return a collate_fn that pads + masks. Mask sentinel = ``mask_id`` (UNK)."""
    pad_id = vocab.pad_id
    eos_id = vocab.eos_id
    bos_id = vocab.bos_id
    vocab_size = len(vocab)
    rng = random.Random()

    def _collate(batch: List[torch.Tensor]):
        from torch.nn.utils.rnn import pad_sequence
        src = pad_sequence(batch, batch_first=True, padding_value=pad_id)   # (B, T)
        labels = torch.full_like(src, -100)
        masked = src.clone()

        for i in range(src.size(0)):
            for j in range(src.size(1)):
                tok = int(src[i, j])
                if tok in (pad_id, eos_id, bos_id):
                    continue
                if rng.random() >= mlm_prob:
                    continue
                labels[i, j] = tok
                r = rng.random()
                if r < 0.8:
                    masked[i, j] = mask_id                                  # [MASK]
                elif r < 0.9:
                    masked[i, j] = rng.randrange(vocab_size)                # random
                # else: keep original
        return masked, labels
    return _collate


# ---------------------------------------------------------------------------
# Vocab + corpus loading
# ---------------------------------------------------------------------------

def _read_corpus(path: str) -> str:
    """Read a corpus file as a single normalized string, one doc per line."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return normalize(text)


def _build_pretrain_vocab(corpus_text: str, lexicon_path: Optional[str]) -> Vocab:
    """Char vocab over the corpus, optionally extended by lexicon characters."""
    char_set = set(corpus_text)
    if lexicon_path:
        for w, _ in load_tsv(lexicon_path):
            char_set.update(w)
    char_set.discard("")
    return Vocab.from_tokens(sorted(char_set), tokenizer=TOKENIZER_CHAR)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pretrain(cfg: PretrainConfig) -> Path:
    """Run masked-character pretraining and return the path to ``pretrain.pt``."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto" else torch.device(cfg.device)
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # --- Corpus + vocab -----------------------------------------------------
    corpus_text = _read_corpus(cfg.corpus_path)
    if not corpus_text:
        raise ValueError(f"corpus at {cfg.corpus_path} is empty after normalize")
    vocab = _build_pretrain_vocab(corpus_text, cfg.lexicon_path)
    vocab.save(out_dir / "src_vocab.json")
    print(f"vocab size = {len(vocab)}  (chars + specials)")

    corpus_ids = vocab.encode(corpus_text)                                  # one big list
    print(f"corpus tokens = {len(corpus_ids):,}")

    ds = _MaskedCharDataset(
        corpus_ids,
        window=min(cfg.window, cfg.max_len),
        stride=cfg.stride,
        min_chars=cfg.min_chars_per_window,
    )
    print(f"windows = {len(ds):,}  (window={cfg.window}, stride={cfg.stride})")

    collate = _make_mask_collate(vocab, cfg.mlm_prob, mask_id=vocab.unk_id)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # --- Model + optimizer --------------------------------------------------
    model = _EncoderOnly(
        vocab_size=len(vocab),
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout, max_len=cfg.max_len,
        pad_id=vocab.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"encoder + MLM head params = {n_params:,}")

    amp_enabled, amp_dtype = _resolve_amp(cfg.amp, device)
    scaler = (
        torch.cuda.amp.GradScaler()
        if (amp_enabled and amp_dtype == torch.float16) else None
    )
    if amp_enabled:
        print(f"AMP: {amp_dtype} on {device}")

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                  betas=(0.9, 0.98), eps=1e-9)

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * cfg.epochs

    # --- Train loop ---------------------------------------------------------
    history: List[dict] = []
    history_path = out_dir / "history.json"
    last_ckpt = out_dir / "pretrain_last.pt"
    best_ckpt = out_dir / "pretrain.pt"
    best_loss = math.inf
    step = 0

    epoch_iter = range(1, cfg.epochs + 1)
    if cfg.progress:
        epoch_iter = _tqdm(list(epoch_iter), desc="pretrain", position=0)

    for epoch in epoch_iter:
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_predicted = 0
        n_batches = 0

        batch_iter = loader
        if cfg.progress:
            batch_iter = _tqdm(loader, desc=f"epoch {epoch}",
                               leave=False, position=1, total=len(loader))

        for masked, labels in batch_iter:
            step += 1
            masked = masked.to(device); labels = labels.to(device)
            lr = _cosine_lr(step, cfg.lr, cfg.warmup_steps, total_steps,
                            cfg.min_lr_ratio)
            for pg in optim.param_groups:
                pg["lr"] = lr

            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with amp_ctx:
                logits = model(masked)                                        # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            with torch.no_grad():
                pred = logits.argmax(-1)
                m = labels != -100
                total_correct += int((pred[m] == labels[m]).sum().item())
                total_predicted += int(m.sum().item())

            total_loss += float(loss.item())
            n_batches += 1
            if cfg.progress and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(total_correct / max(1, total_predicted)):.3f}",
                    lr=f"{lr:.1e}",
                )

        train_loss = total_loss / max(1, n_batches)
        masked_acc = total_correct / max(1, total_predicted)
        dt = time.time() - t0
        history.append({
            "epoch": int(epoch), "train_loss": float(train_loss),
            "masked_acc": float(masked_acc), "lr": float(lr), "seconds": float(dt),
        })
        try:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception:
            pass

        msg = (f"[ep {epoch:3d}] train_loss={train_loss:.4f}  "
               f"masked_acc={masked_acc:.3f}  ({dt:.1f}s)")
        if cfg.progress and hasattr(epoch_iter, "write"):
            epoch_iter.write(msg)
        else:
            print(msg)

        # Save state. Keys mirror G2PTransformer's encoder so the supervised
        # loader (train._load_pretrained_encoder) finds them with the standard
        # ``src_embed.``, ``src_pos.``, ``transformer.encoder.`` prefixes.
        payload = {
            "model_state": {k: v.detach().cpu()
                            for k, v in model.state_dict().items()},
            "src_vocab": vocab.to_dict(),
            "config": {
                "d_model": cfg.d_model, "nhead": cfg.nhead,
                "num_encoder_layers": cfg.num_encoder_layers,
                "dim_feedforward": cfg.dim_feedforward,
                "dropout": cfg.dropout, "max_len": cfg.max_len,
                "pad_id": vocab.pad_id, "mlm_prob": cfg.mlm_prob,
            },
            "epoch": epoch, "train_loss": train_loss, "masked_acc": masked_acc,
        }
        torch.save(payload, last_ckpt)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(payload, best_ckpt)

    print(f"\nBest pretraining loss = {best_loss:.4f}  → {best_ckpt}")
    return best_ckpt
