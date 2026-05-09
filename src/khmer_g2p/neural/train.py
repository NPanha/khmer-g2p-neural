"""Training loop for the character-level Khmer G2P Transformer."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from khmer_g2p.lexicon import load_tsv
from khmer_g2p.neural.aux_labels import IGNORE_INDEX, extract_aux_labels
from khmer_g2p.neural.dataset import G2PDataset, make_collate, split_pairs
from khmer_g2p.neural.ema import ModelEma
from khmer_g2p.neural.metrics import per, wer
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.vocab import Vocab, build_vocabs


# Encoder-side parameter prefixes that get warm-started from a pretrain ckpt.
_ENCODER_PREFIXES: Tuple[str, ...] = (
    "src_embed.", "src_pos.", "transformer.encoder.",
)


# tqdm is optional — fall back to a no-op if not installed.
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(it=None, *_, **__):
        return it if it is not None else iter([])


@dataclass
class TrainConfig:
    data_path: str
    out_dir: str = "checkpoints"
    # Model — v0.4 defaults: asymmetric 6-encoder / 3-decoder.
    # Alignment is the harder half of G2P, so we spend more depth on the encoder.
    d_model: int = 384
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 3
    dim_feedforward: int = 1536
    dropout: float = 0.3
    max_len: int = 256
    # Tie decoder input embedding and output projection. Saves params and
    # routinely wins 0.3-0.8 PER on this scale of data.
    tie_embeddings: bool = True
    # Training
    batch_size: int = 64
    epochs: int = 80
    lr: float = 5e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    # LR schedule. "cosine" is the v0.4 default — cosine decay from peak_lr
    # at the end of warmup down to ``peak_lr * min_lr_ratio``. "noam" is the
    # original Transformer/G2P schedule kept for backwards compatibility.
    lr_schedule: str = "cosine"   # "cosine" | "noam"
    min_lr_ratio: float = 0.05    # cosine floor as a fraction of peak_lr
    # Splits
    val_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 42
    # Runtime
    device: str = "auto"   # "auto" | "cpu" | "cuda"
    num_workers: int = 0
    log_every: int = 50
    # Early stopping
    patience: int = 12
    # Tokenization: "char" (default, back-compat) or "phoneme" (IPA phonemes)
    tgt_tokenizer: str = "phoneme"
    # tqdm progress bars (set False for CLI / CI logs)
    progress: bool = True
    # CTC auxiliary loss on the encoder (regularizes alignment).
    # loss = CE(decoder) + ctc_weight * CTC(encoder)
    use_ctc: bool = True
    ctc_weight: float = 0.3
    # Exponential moving average of weights. 0 disables; 0.999 is a strong
    # default for batch-size 64. Eval / best.pt use the EMA weights.
    ema_decay: float = 0.999
    # Mixed-precision training. Only effective on CUDA; "auto" = bf16 on Ampere+
    # else fp16. "off" disables AMP entirely.
    amp: str = "auto"             # "auto" | "off" | "fp16" | "bf16"
    # 0.5: encoder-side auxiliary objectives. Weights of 0 disable.
    series_weight: float = 0.0    # CE on per-position consonant-series labels
    syllable_weight: float = 0.0  # CE on per-position syllable-boundary labels
    # 0.5: optional warm-start from a masked-character pretraining checkpoint.
    # If given, the model's src_vocab is forced to the pretrain vocab and the
    # encoder weights (incl. src_embed + src_pos) are copied in before training.
    pretrained_encoder: Optional[str] = None


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _lr_schedule(step: int, warmup: int, d_model: int) -> float:
    """Noam-style: linear warmup then inverse-sqrt decay (normalized by d_model)."""
    step = max(1, step)
    warmup = max(1, warmup)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


def _cosine_lr(
    step: int,
    peak_lr: float,
    warmup: int,
    total: int,
    min_ratio: float = 0.05,
) -> float:
    """Linear warmup to peak_lr, then cosine decay to ``peak_lr * min_ratio``.

    ``step`` is 1-indexed. Returns the absolute LR (not a multiplier).
    """
    step = max(1, step)
    warmup = max(1, warmup)
    total = max(warmup + 1, total)
    if step <= warmup:
        return peak_lr * (step / warmup)
    progress = (step - warmup) / (total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cos)


def _load_pretrained_encoder(model: nn.Module, pretrain_state: dict) -> int:
    """Copy encoder-side weights from a pretraining state_dict into ``model``.

    Only keys whose name starts with one of ``_ENCODER_PREFIXES`` are copied —
    decoder, output projection, CTC head, and aux heads stay at their fresh
    Xavier init. Returns the number of tensors actually transferred.

    Shape mismatches raise loudly: the supervised model's encoder
    architecture must match the pretrained one.
    """
    target_state = model.state_dict()
    n_loaded = 0
    for k, v in pretrain_state.items():
        if not any(k.startswith(p) for p in _ENCODER_PREFIXES):
            continue
        if k not in target_state:
            continue
        if target_state[k].shape != v.shape:
            raise RuntimeError(
                f"pretrained encoder shape mismatch on '{k}': "
                f"target {tuple(target_state[k].shape)} vs "
                f"pretrain {tuple(v.shape)}"
            )
        target_state[k].copy_(v.to(target_state[k].dtype).to(target_state[k].device))
        n_loaded += 1
    return n_loaded


def _resolve_amp(mode: str, device: torch.device):
    """Return ``(enabled, dtype)`` for the autocast context.

    Mode ``"auto"`` picks bf16 when the GPU advertises bf16 support, otherwise
    fp16. AMP is silently disabled on non-CUDA devices.
    """
    if mode == "off" or device.type != "cuda":
        return False, None
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    # auto
    bf16_ok = False
    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    return True, (torch.bfloat16 if bf16_ok else torch.float16)


def _build_ctc_targets(
    tgt_out: torch.Tensor,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten tgt_out rows into a 1-D CTC target tensor, stripping BOS/EOS/PAD.

    Returns:
        flat_targets  (sum(L_i),) long tensor of target token ids.
        target_lens   (B,)        long tensor of per-row lengths.
    """
    flat: List[int] = []
    lens: List[int] = []
    specials = {pad_id, bos_id, eos_id}
    for row in tgt_out.tolist():
        clean = [t for t in row if t not in specials]
        flat.extend(clean)
        lens.append(len(clean))
    device = tgt_out.device
    flat_t = torch.tensor(flat, dtype=torch.long, device=device)
    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    return flat_t, lens_t


# ---------------------------------------------------------------------------
# Greedy decode (for eval during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: G2PTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    words: List[str],
    device: torch.device,
    max_len: int = 128,
) -> List[str]:
    model.eval()
    outs: List[str] = []
    for word in words:
        src_ids = src_vocab.encode(word, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory, src_kpm = model.encode(src)
        ys = torch.tensor([[tgt_vocab.bos_id]], dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = model.decode(ys, memory, src_kpm)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == tgt_vocab.eos_id:
                break
        outs.append(tgt_vocab.decode(ys[0].tolist()[1:]))
    return outs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> Path:
    """Train a G2P Transformer. Returns path to the best checkpoint."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    # --- Data ----------------------------------------------------------------
    pairs = load_tsv(cfg.data_path)
    if len(pairs) < 10:
        raise ValueError(
            f"Only {len(pairs)} entries in {cfg.data_path}. Training a neural "
            "model needs at least a few hundred; ideally thousands."
        )
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    print(f"Loaded {len(pairs)} entries → "
          f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")

    # If a pretrained encoder was passed, its src_vocab is the source of
    # truth (so token IDs line up). Otherwise build vocabs from training pairs.
    pretrain_payload = None
    if cfg.pretrained_encoder:
        pretrain_payload = torch.load(
            cfg.pretrained_encoder, map_location="cpu", weights_only=False
        )
        src_vocab = Vocab.from_dict(pretrain_payload["src_vocab"])
        # Target vocab is still derived from the lexicon — pretraining is
        # source-only.
        _, tgt_vocab = build_vocabs(
            train_pairs,
            src_tokenizer="char",
            tgt_tokenizer=cfg.tgt_tokenizer,
        )
        print(f"warm-start: loaded pretrain src_vocab "
              f"({len(src_vocab)} symbols) from {cfg.pretrained_encoder}")
    else:
        src_vocab, tgt_vocab = build_vocabs(
            train_pairs,
            src_tokenizer="char",
            tgt_tokenizer=cfg.tgt_tokenizer,
        )
    src_vocab.save(out_dir / "src_vocab.json")
    tgt_vocab.save(out_dir / "tgt_vocab.json")
    print(f"Vocab sizes: src={len(src_vocab)} (char) "
          f"tgt={len(tgt_vocab)} ({cfg.tgt_tokenizer})")

    train_ds = G2PDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = G2PDataset(val_pairs, src_vocab, tgt_vocab) if val_pairs else None
    test_ds = G2PDataset(test_pairs, src_vocab, tgt_vocab) if test_pairs else None

    collate = make_collate(src_vocab.pad_id, tgt_vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # --- Model ---------------------------------------------------------------
    mcfg = G2PConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        pad_id=src_vocab.pad_id,
        use_ctc=cfg.use_ctc,
        tie_embeddings=cfg.tie_embeddings,
        use_series_head=cfg.series_weight > 0.0,
        use_syllable_head=cfg.syllable_weight > 0.0,
    )
    model = G2PTransformer(mcfg).to(device)

    # Warm-start the encoder from a masked-character pretraining checkpoint.
    if pretrain_payload is not None:
        loaded = _load_pretrained_encoder(model, pretrain_payload["model_state"])
        print(f"warm-start: copied {loaded} encoder tensors from pretrain")

    extras: list = []
    if cfg.use_ctc:
        extras.append(f"+CTC aux w={cfg.ctc_weight}")
    if cfg.tie_embeddings:
        extras.append("tied-emb")
    if mcfg.use_series_head:
        extras.append(f"+series w={cfg.series_weight}")
    if mcfg.use_syllable_head:
        extras.append(f"+syll w={cfg.syllable_weight}")
    print(f"Model params: {model.num_parameters():,}"
          + (f"  ({', '.join(extras)})" if extras else ""))

    # --- AMP setup ----------------------------------------------------------
    amp_enabled, amp_dtype = _resolve_amp(cfg.amp, device)
    scaler = (
        torch.cuda.amp.GradScaler()
        if (amp_enabled and amp_dtype == torch.float16)
        else None
    )
    if amp_enabled:
        print(f"AMP: {amp_dtype} on {device}"
              + ("  (+GradScaler)" if scaler is not None else ""))

    # --- EMA setup ----------------------------------------------------------
    ema = ModelEma(model, decay=cfg.ema_decay) if cfg.ema_decay > 0.0 else None
    if ema is not None:
        print(f"Weight EMA enabled (target decay={cfg.ema_decay})")

    # --- Optimizer & loss ----------------------------------------------------
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                  betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_id,
        label_smoothing=cfg.label_smoothing,
    )

    # Total optimizer steps for the cosine schedule.
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.lr_schedule == "cosine":
        print(f"LR schedule: cosine, warmup={cfg.warmup_steps}, "
              f"total_steps={total_steps}, min_ratio={cfg.min_lr_ratio}")
    else:
        print(f"LR schedule: noam, warmup={cfg.warmup_steps}")

    def _lr_for_step(step_idx: int) -> float:
        if cfg.lr_schedule == "cosine":
            return _cosine_lr(
                step_idx, cfg.lr, cfg.warmup_steps, total_steps, cfg.min_lr_ratio
            )
        # noam — same normalized form as before so external behavior matches.
        ratio = _lr_schedule(step_idx, cfg.warmup_steps, cfg.d_model)
        peak_ratio = _lr_schedule(max(1, cfg.warmup_steps), cfg.warmup_steps, cfg.d_model)
        return cfg.lr * ratio / peak_ratio

    # --- Training loop -------------------------------------------------------
    best_val_per = math.inf
    best_ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"
    history_path = out_dir / "history.json"
    history: List[dict] = []
    patience_left = cfg.patience
    step = 0

    epoch_iter = range(1, cfg.epochs + 1)
    if cfg.progress:
        epoch_iter = _tqdm(list(epoch_iter), desc="epochs", position=0)

    for epoch in epoch_iter:
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        batch_iter = train_loader
        if cfg.progress:
            batch_iter = _tqdm(train_loader, desc=f"epoch {epoch}", leave=False,
                               position=1, total=len(train_loader))

        for src, tgt_in, tgt_out in batch_iter:
            step += 1
            src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)

            # LR schedule (cosine-with-warmup or Noam, depending on cfg).
            lr = _lr_for_step(step)
            for pg in optim.param_groups:
                pg["lr"] = lr

            # AMP autocast context (no-op when amp_enabled is False).
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with amp_ctx:
                # Encoder → cross-entropy on decoder logits.
                memory, src_kpm = model.encode(src)
                logits = model.decode(tgt_in, memory, src_kpm)
                ce_loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
                )

                # Optional CTC auxiliary loss on the encoder memory.
                ctc_val = 0.0
                if cfg.use_ctc and model.ctc_proj is not None:
                    # CTC needs fp32 log-probs for numerical stability under AMP.
                    with torch.autocast(device_type=device.type, enabled=False):
                        ctc_log_probs = model.ctc_log_probs(memory.float())
                        input_lens = (src != src_vocab.pad_id).sum(dim=1).to(torch.long)
                        flat_targets, target_lens = _build_ctc_targets(
                            tgt_out,
                            pad_id=tgt_vocab.pad_id,
                            bos_id=tgt_vocab.bos_id,
                            eos_id=tgt_vocab.eos_id,
                        )
                        if target_lens.numel() > 0 and target_lens.sum().item() > 0:
                            ctc_loss = F.ctc_loss(
                                ctc_log_probs,
                                flat_targets,
                                input_lens,
                                target_lens,
                                blank=tgt_vocab.pad_id,
                                reduction="mean",
                                zero_infinity=True,
                            )
                            loss = ce_loss + cfg.ctc_weight * ctc_loss
                            ctc_val = float(ctc_loss.item())
                        else:
                            loss = ce_loss
                else:
                    loss = ce_loss

                # 0.5: encoder-side structural aux losses. Labels are derived
                # on the fly from src + src_vocab — no dataset changes needed.
                series_val = 0.0
                syllable_val = 0.0
                if model.series_head is not None or model.syllable_head is not None:
                    with torch.autocast(device_type=device.type, enabled=False):
                        if model.series_head is not None:
                            series_lbl = extract_aux_labels(src, src_vocab)[0]
                            ser_logits = model.series_logits(memory.float())
                            ser_loss = F.cross_entropy(
                                ser_logits.reshape(-1, ser_logits.size(-1)),
                                series_lbl.reshape(-1),
                                ignore_index=IGNORE_INDEX,
                            )
                            if torch.isfinite(ser_loss):
                                loss = loss + cfg.series_weight * ser_loss
                                series_val = float(ser_loss.item())
                        if model.syllable_head is not None:
                            syll_lbl = extract_aux_labels(src, src_vocab)[1]
                            sy_logits = model.syllable_logits(memory.float())
                            sy_loss = F.cross_entropy(
                                sy_logits.reshape(-1, sy_logits.size(-1)),
                                syll_lbl.reshape(-1),
                                ignore_index=IGNORE_INDEX,
                            )
                            if torch.isfinite(sy_loss):
                                loss = loss + cfg.syllable_weight * sy_loss
                                syllable_val = float(sy_loss.item())

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            # Update EMA shadow after the optimizer step.
            if ema is not None:
                ema.update(model, step=step)

            total_loss += loss.item()
            n_batches += 1
            if cfg.progress and hasattr(batch_iter, "set_postfix"):
                if cfg.use_ctc:
                    batch_iter.set_postfix(
                        loss=f"{loss.item():.4f}",
                        ce=f"{ce_loss.item():.3f}",
                        ctc=f"{ctc_val:.3f}",
                        lr=f"{lr:.2e}",
                    )
                else:
                    batch_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
            elif step % cfg.log_every == 0 and not cfg.progress:
                print(f"  step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}")

        train_loss = total_loss / max(1, n_batches)
        dt = time.time() - t0

        # --- Eval ------------------------------------------------------------
        # Eval on EMA weights when available; the live model's late-stage
        # zigzag is exactly what the EMA is meant to filter out.
        val_per_score = float("nan")
        val_wer_score = float("nan")
        if val_ds is not None and len(val_ds) > 0:
            val_words = [w for w, _ in val_pairs]
            val_refs = [p for _, p in val_pairs]
            ema_ctx = ema.apply_to(model) if ema is not None else nullcontext()
            with ema_ctx:
                preds = greedy_decode(model, src_vocab, tgt_vocab, val_words, device,
                                      max_len=cfg.max_len)
            val_per_score = per(preds, val_refs)
            val_wer_score = wer(preds, val_refs)

        # Record history for plotting.
        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_per": float(val_per_score),
            "val_wer": float(val_wer_score),
            "lr": float(lr),
            "seconds": float(dt),
        })
        try:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception:
            pass  # non-fatal — don't crash training on a disk hiccup

        msg = (f"[epoch {epoch:3d}] train_loss={train_loss:.4f} "
               f"val_PER={val_per_score:.4f} val_WER={val_wer_score:.4f} "
               f"({dt:.1f}s)")
        if cfg.progress and hasattr(epoch_iter, "write"):
            epoch_iter.write(msg)
        else:
            print(msg)

        # --- Checkpoint ------------------------------------------------------
        # last.pt always carries the live model so training can be resumed.
        # best.pt carries the EMA weights when EMA is on (those are the ones
        # we evaluated above and the ones that should be deployed).
        live_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        last_payload = {
            "model_state": live_state,
            "config": mcfg.to_dict(),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_per": val_per_score,
            "val_wer": val_wer_score,
            "ema": ema.state_dict() if ema is not None else None,
        }
        torch.save(last_payload, last_ckpt_path)

        improved = val_per_score < best_val_per
        if improved:
            best_val_per = val_per_score
            patience_left = cfg.patience
            best_payload = dict(last_payload)
            if ema is not None:
                # The deployed weights *are* the EMA — write them as the
                # primary model_state so NeuralG2P.from_checkpoint just works.
                best_payload["model_state"] = ema.state_dict()
                best_payload["from_ema"] = True
            torch.save(best_payload, best_ckpt_path)
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

    # --- Final test eval -----------------------------------------------------
    if test_ds is not None and len(test_ds) > 0:
        print("\nLoading best checkpoint for test evaluation…")
        payload = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        # Cast EMA's float32 state back to the model's parameter dtype so
        # load_state_dict doesn't change parameter types under us.
        target_dtypes = {k: v.dtype for k, v in model.state_dict().items()}
        casted_state = {
            k: (v.to(target_dtypes[k]) if (k in target_dtypes and v.is_floating_point())
                else v)
            for k, v in payload["model_state"].items()
        }
        model.load_state_dict(casted_state)
        model.tie_weights_if_configured()
        test_words = [w for w, _ in test_pairs]
        test_refs = [p for _, p in test_pairs]
        preds = greedy_decode(model, src_vocab, tgt_vocab, test_words, device,
                              max_len=cfg.max_len)
        t_per = per(preds, test_refs)
        t_wer = wer(preds, test_refs)
        print(f"TEST  PER={t_per:.4f}  WER={t_wer:.4f}  "
              f"(n={len(test_pairs)})")
        (out_dir / "test_metrics.json").write_text(
            json.dumps({"per": t_per, "wer": t_wer, "n": len(test_pairs)}, indent=2),
            encoding="utf-8",
        )

    return best_ckpt_path
