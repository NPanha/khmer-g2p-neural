"""Low-LR resume fine-tuning.

After the main training run plateaus (Noam decay → tiny effective LR, dropout
0.3 capping the late-stage ceiling), resuming from ``best.pt`` with a flat
small LR, lower dropout, and CTC turned off typically buys another 1–3
absolute PER points on Khmer G2P.

The recipe this module implements:

* Load model + vocabs from an existing checkpoint.
* Override config-time values that we want to relax: dropout (default 0.1),
  CTC weight (default 0.0 — head still loaded but unused).
* Use a flat LR (default 5e-5) with no warmup and no Noam decay.
* Patience halved (default 6) since we expect convergence to be quick.
* Save to a *new* output directory so the original best.pt stays untouched.

Public entry point: :func:`finetune` — accepts a :class:`FinetuneConfig`,
loads a checkpoint, fine-tunes, and writes a new ``best.pt``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from khmer_g2p.lexicon import load_tsv
from khmer_g2p.neural.dataset import G2PDataset, make_collate, split_pairs
from khmer_g2p.neural.metrics import per, wer
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.train import greedy_decode  # reuse eval decoder
from khmer_g2p.neural.vocab import Vocab


try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(it=None, *_, **__):
        return it if it is not None else iter([])


@dataclass
class FinetuneConfig:
    ckpt_path: str                           # existing best.pt
    data_path: str                           # same TSV as training
    out_dir: str = "checkpoints_finetune"

    # Fine-tune hyperparameters
    lr: float = 5e-5
    dropout: float = 0.1                     # relax from 0.3
    epochs: int = 20
    patience: int = 6
    batch_size: int = 64
    weight_decay: float = 0.0                # decoupled fine-tune; turn wd off
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    ctc_weight: float = 0.0                  # disable CTC during fine-tune
    seed: int = 42                           # used only for the *finetune* split

    # Use the *same* split as training by passing the same seed/fractions.
    val_frac: float = 0.05
    test_frac: float = 0.05

    # Runtime
    device: str = "auto"
    num_workers: int = 0
    progress: bool = True


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_dropout(module: nn.Module, p: float) -> int:
    """Walk the module tree and set ``p`` on every nn.Dropout in place.

    Returns the number of dropout modules updated. We also overwrite
    each transformer layer's ``self.dropout`` scalar attribute when present
    (PyTorch's nn.Transformer stores it for the residual paths).
    """
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            n += 1
        # nn.TransformerEncoderLayer / nn.TransformerDecoderLayer carry a
        # plain float attribute named `dropout` separate from their dropout
        # submodules — keep it consistent.
        if hasattr(m, "dropout") and isinstance(getattr(m, "dropout"), float):
            try:
                m.dropout = p
            except (AttributeError, TypeError):
                pass
    return n


def finetune(cfg: FinetuneConfig) -> Path:
    """Resume fine-tuning from an existing checkpoint and return the new best.pt path."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    # --- Load checkpoint ----------------------------------------------------
    payload = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    mcfg = G2PConfig.from_dict(payload["config"])
    src_vocab = Vocab.from_dict(payload["src_vocab"])
    tgt_vocab = Vocab.from_dict(payload["tgt_vocab"])

    model = G2PTransformer(mcfg).to(device)
    model.load_state_dict(payload["model_state"])
    n_dropout_modules = _set_dropout(model, cfg.dropout)
    print(f"Loaded {cfg.ckpt_path} → params={model.num_parameters():,}")
    print(f"Relaxed dropout to {cfg.dropout} on {n_dropout_modules} modules")

    # Save the (potentially relaxed) vocab back into the new out dir for
    # parity with the regular train.py output layout.
    src_vocab.save(out_dir / "src_vocab.json")
    tgt_vocab.save(out_dir / "tgt_vocab.json")

    # --- Data ---------------------------------------------------------------
    pairs = load_tsv(cfg.data_path)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    print(f"Loaded {len(pairs)} entries → "
          f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")

    train_ds = G2PDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = G2PDataset(val_pairs, src_vocab, tgt_vocab) if val_pairs else None

    collate = make_collate(src_vocab.pad_id, tgt_vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # --- Optimizer ----------------------------------------------------------
    # Flat LR — no Noam decay, no warmup. Fine-tuning is short enough that the
    # late-stage refinement we wanted is exactly what a constant small LR gives.
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                  betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_id,
        label_smoothing=cfg.label_smoothing,
    )

    # --- Initial val score so we know our baseline -------------------------
    initial_per = float("nan")
    initial_wer = float("nan")
    if val_ds is not None and len(val_ds) > 0:
        val_words = [w for w, _ in val_pairs]
        val_refs = [p for _, p in val_pairs]
        preds = greedy_decode(model, src_vocab, tgt_vocab, val_words, device,
                              max_len=mcfg.max_len)
        initial_per = per(preds, val_refs)
        initial_wer = wer(preds, val_refs)
        print(f"baseline (loaded checkpoint)  val_PER={initial_per:.4f}  "
              f"val_WER={initial_wer:.4f}")

    # --- Loop ---------------------------------------------------------------
    best_val_per = initial_per if not math.isnan(initial_per) else math.inf
    best_ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"
    history_path = out_dir / "history.json"
    history: List[dict] = []
    patience_left = cfg.patience

    epoch_iter = range(1, cfg.epochs + 1)
    if cfg.progress:
        epoch_iter = _tqdm(list(epoch_iter), desc="finetune", position=0)

    for epoch in epoch_iter:
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        batch_iter = train_loader
        if cfg.progress:
            batch_iter = _tqdm(train_loader, desc=f"ft epoch {epoch}",
                               leave=False, position=1, total=len(train_loader))

        for src, tgt_in, tgt_out in batch_iter:
            src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)

            memory, src_kpm = model.encode(src)
            logits = model.decode(tgt_in, memory, src_kpm)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            )

            # CTC is disabled by default during fine-tune (ctc_weight=0).
            # If you really want to keep it, set ctc_weight > 0 in the config.
            if cfg.ctc_weight > 0 and model.ctc_proj is not None:
                ctc_log_probs = model.ctc_log_probs(memory)
                input_lens = (src != src_vocab.pad_id).sum(dim=1).to(torch.long)
                # Reuse the same flattening as train.py (BOS/EOS/PAD stripped).
                specials = {tgt_vocab.pad_id, tgt_vocab.bos_id, tgt_vocab.eos_id}
                flat: list = []
                lens: list = []
                for row in tgt_out.tolist():
                    clean = [t for t in row if t not in specials]
                    flat.extend(clean)
                    lens.append(len(clean))
                if sum(lens) > 0:
                    flat_t = torch.tensor(flat, dtype=torch.long, device=device)
                    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
                    ctc_loss = torch.nn.functional.ctc_loss(
                        ctc_log_probs, flat_t, input_lens, lens_t,
                        blank=tgt_vocab.pad_id, reduction="mean", zero_infinity=True,
                    )
                    loss = loss + cfg.ctc_weight * ctc_loss

            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            total_loss += loss.item()
            n_batches += 1
            if cfg.progress and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cfg.lr:.1e}")

        train_loss = total_loss / max(1, n_batches)
        dt = time.time() - t0

        # --- Eval -----------------------------------------------------------
        val_per_score = float("nan")
        val_wer_score = float("nan")
        if val_ds is not None and len(val_ds) > 0:
            val_words = [w for w, _ in val_pairs]
            val_refs = [p for _, p in val_pairs]
            preds = greedy_decode(model, src_vocab, tgt_vocab, val_words, device,
                                  max_len=mcfg.max_len)
            val_per_score = per(preds, val_refs)
            val_wer_score = wer(preds, val_refs)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_per": float(val_per_score),
            "val_wer": float(val_wer_score),
            "lr": float(cfg.lr),
            "seconds": float(dt),
        })
        try:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception:
            pass

        msg = (f"[ft epoch {epoch:3d}] train_loss={train_loss:.4f} "
               f"val_PER={val_per_score:.4f} val_WER={val_wer_score:.4f} ({dt:.1f}s)")
        if cfg.progress and hasattr(epoch_iter, "write"):
            epoch_iter.write(msg)
        else:
            print(msg)

        # --- Checkpoint -----------------------------------------------------
        ckpt_payload = {
            "model_state": model.state_dict(),
            "config": mcfg.to_dict(),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_per": val_per_score,
            "val_wer": val_wer_score,
            "finetune": True,
            "finetune_from": cfg.ckpt_path,
        }
        torch.save(ckpt_payload, last_ckpt_path)

        improved = val_per_score < best_val_per
        if improved:
            best_val_per = val_per_score
            patience_left = cfg.patience
            torch.save(ckpt_payload, best_ckpt_path)
            info = f"  ✓ new best val_PER={val_per_score:.4f} → {best_ckpt_path}"
            if cfg.progress and hasattr(epoch_iter, "write"):
                epoch_iter.write(info)
            else:
                print(info)
        else:
            patience_left -= 1
            if patience_left <= 0:
                info = f"  early stop: no improvement for {cfg.patience} epochs"
                if cfg.progress and hasattr(epoch_iter, "write"):
                    epoch_iter.write(info)
                else:
                    print(info)
                break

    # If we never saved a new best (rare but possible), copy the loaded
    # weights into best.pt so downstream code can always load this dir.
    if not best_ckpt_path.exists():
        torch.save(ckpt_payload, best_ckpt_path)

    print(f"\nDone. baseline_PER={initial_per:.4f}  best_PER={best_val_per:.4f}  "
          f"Δ={initial_per - best_val_per:+.4f}")
    return best_ckpt_path
