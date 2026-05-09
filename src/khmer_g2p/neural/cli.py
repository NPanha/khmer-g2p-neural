"""Command-line entry points for neural training and prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# khmer-g2p-train
# ---------------------------------------------------------------------------

def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khmer-g2p-train",
        description="Train a character-level Transformer G2P on a Khmer lexicon.",
    )
    p.add_argument("--data", required=True, type=Path,
                   help="Path to TSV lexicon (word<TAB>ipa; header row OK).")
    p.add_argument("--out", default="checkpoints", type=Path,
                   help="Directory to write checkpoints + vocabs.")
    # Model — v0.4 defaults: asymmetric 6-encoder / 3-decoder, tied embeddings.
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ffn", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--tie-embeddings", dest="tie_embeddings",
                   action="store_true", default=True,
                   help="Tie decoder input embedding with output projection (default: on).")
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false",
                   help="Disable embedding tying (matches the v0.3 architecture).")
    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=12)
    # LR schedule + EMA + AMP
    p.add_argument("--lr-schedule", choices=["cosine", "noam"], default="cosine",
                   help="LR schedule. 'cosine' is the v0.4 default.")
    p.add_argument("--min-lr-ratio", type=float, default=0.05,
                   help="Cosine floor as a fraction of peak LR (default: 0.05).")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Exponential moving average of weights. 0 disables.")
    p.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto",
                   help="Mixed-precision training. 'auto' picks bf16 when supported.")
    # 0.5: aux objectives + pretrained-encoder warm start.
    p.add_argument("--series-weight", type=float, default=0.0,
                   help="Weight on the encoder's per-position consonant-series "
                        "head. 0 disables (default). 0.1-0.3 is a sensible range.")
    p.add_argument("--syllable-weight", type=float, default=0.0,
                   help="Weight on the encoder's per-position syllable-boundary "
                        "head. 0 disables (default). 0.1-0.3 is a sensible range.")
    p.add_argument("--pretrained-encoder", type=Path, default=None,
                   help="Path to a pretrain.pt produced by khmer-g2p-pretrain. "
                        "Source vocab is taken from the pretrain checkpoint and "
                        "encoder weights are warm-started before supervised "
                        "training begins.")
    p.add_argument("--tgt-tokenizer", choices=["char", "phoneme"], default="phoneme",
                   help="How to tokenize the IPA target. 'phoneme' glues "
                        "length/aspiration marks into their base phoneme, "
                        "shortening target sequences and usually improving "
                        "PER by several points.")
    # CTC auxiliary loss on the encoder.
    p.add_argument("--ctc", dest="use_ctc", action="store_true", default=True,
                   help="Enable CTC auxiliary loss on encoder (default: on).")
    p.add_argument("--no-ctc", dest="use_ctc", action="store_false",
                   help="Disable CTC auxiliary loss.")
    p.add_argument("--ctc-weight", type=float, default=0.3,
                   help="Weight of CTC loss in total loss. "
                        "loss = CE + ctc_weight * CTC. (default: 0.3)")
    # Splits
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # Runtime
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    return p


def train_main(argv: List[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    # Import here so `khmer-g2p-train --help` works without torch.
    from khmer_g2p.neural.train import TrainConfig, train

    cfg = TrainConfig(
        data_path=str(args.data),
        out_dir=str(args.out),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_len=args.max_len,
        tie_embeddings=args.tie_embeddings,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        patience=args.patience,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay=args.ema_decay,
        amp=args.amp,
        series_weight=args.series_weight,
        syllable_weight=args.syllable_weight,
        pretrained_encoder=str(args.pretrained_encoder) if args.pretrained_encoder else None,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
        log_every=args.log_every,
        tgt_tokenizer=args.tgt_tokenizer,
        use_ctc=args.use_ctc,
        ctc_weight=args.ctc_weight,
    )
    best = train(cfg)
    print(f"\nBest checkpoint: {best}")
    return 0


# ---------------------------------------------------------------------------
# khmer-g2p-predict
# ---------------------------------------------------------------------------

def build_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khmer-g2p-predict",
        description="Run a trained Khmer G2P Transformer on words.",
    )
    p.add_argument("--ckpt", type=Path,
                   help="Path to a single .pt checkpoint. Mutually exclusive with --ensemble.")
    p.add_argument("--ensemble", type=Path, nargs="+", default=None,
                   help="Two or more checkpoint paths to ensemble (logit averaging). "
                        "Mutually exclusive with --ckpt.")
    p.add_argument("text", nargs="?", help="Khmer word. Omit to read stdin.")
    p.add_argument("-f", "--file", type=Path,
                   help="Read one word per line from file.")
    p.add_argument("--batch", action="store_true",
                   help="Treat each line of stdin/file as one word.")
    p.add_argument("--beam", type=int, default=1,
                   help="Beam size (1 = greedy).")
    p.add_argument("--length-penalty", type=float, default=0.6,
                   help="GNMT length penalty α for beam search "
                        "(0 = raw log-prob, default 0.6).")
    p.add_argument("--lexicon", type=Path, default=None,
                   help="Optional TSV lexicon. If given, hybrid lexicon-first "
                        "lookup is used: known words return the canonical IPA, "
                        "unknown words fall through to the neural model.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def predict_main(argv: List[str] | None = None) -> int:
    args = build_predict_parser().parse_args(argv)
    if (args.ckpt is None) == (args.ensemble is None):
        sys.stderr.write("error: pass exactly one of --ckpt or --ensemble.\n")
        return 2

    device = None if args.device == "auto" else args.device
    # Three branches:
    #   --ensemble + --lexicon   → HybridG2P over an EnsembleG2P
    #   --ensemble               → EnsembleG2P
    #   --ckpt    + --lexicon    → HybridG2P over a NeuralG2P
    #   --ckpt                   → NeuralG2P
    if args.ensemble is not None:
        from khmer_g2p.neural.ensemble import EnsembleG2P
        base = EnsembleG2P.from_checkpoints(args.ensemble, device=device)
    else:
        from khmer_g2p.neural.infer import NeuralG2P
        base = NeuralG2P.from_checkpoint(args.ckpt, device=device)

    if args.lexicon is not None:
        from khmer_g2p.hybrid import HybridG2P
        from khmer_g2p.lexicon import Lexicon, load_tsv
        g2p = HybridG2P(base, Lexicon(load_tsv(args.lexicon)))
    else:
        g2p = base

    # Gather input
    if args.text is not None:
        inputs = [args.text]
    elif args.file:
        inputs = [ln for ln in args.file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        data = sys.stdin.read()
        if args.batch:
            inputs = [ln for ln in data.splitlines() if ln.strip()]
        else:
            inputs = [data.strip()]

    for w in inputs:
        print(g2p.convert(w, beam=args.beam, length_penalty=args.length_penalty))
    return 0


# ---------------------------------------------------------------------------
# khmer-g2p-finetune
# ---------------------------------------------------------------------------

def build_finetune_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khmer-g2p-finetune",
        description="Resume fine-tuning from an existing checkpoint with a "
                    "flat low LR, lower dropout, and CTC disabled.",
    )
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to an existing best.pt to resume from.")
    p.add_argument("--data", required=True, type=Path,
                   help="Same TSV lexicon as training (for splits).")
    p.add_argument("--out", default="checkpoints_finetune", type=Path,
                   help="New output directory (kept separate from main run).")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Flat learning rate (no warmup, no Noam decay).")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="New dropout to apply to all dropout submodules.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ctc-weight", type=float, default=0.0,
                   help="Set > 0 to keep CTC during fine-tune (default: off).")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    return p


def finetune_main(argv: List[str] | None = None) -> int:
    args = build_finetune_parser().parse_args(argv)
    from khmer_g2p.neural.finetune import FinetuneConfig, finetune

    cfg = FinetuneConfig(
        ckpt_path=str(args.ckpt),
        data_path=str(args.data),
        out_dir=str(args.out),
        lr=args.lr,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        ctc_weight=args.ctc_weight,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
    )
    best = finetune(cfg)
    print(f"\nBest fine-tuned checkpoint: {best}")
    return 0


# ---------------------------------------------------------------------------
# khmer-g2p-pretrain
# ---------------------------------------------------------------------------

def build_pretrain_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khmer-g2p-pretrain",
        description="Masked-character pretraining of the Khmer encoder on a "
                    "monolingual text corpus.",
    )
    p.add_argument("--corpus", required=True, type=Path,
                   help="Plain-text Khmer corpus (one document per line).")
    p.add_argument("--lexicon", type=Path, default=None,
                   help="Optional lexicon TSV — its characters are added to the "
                        "vocab so supervised training can transfer cleanly.")
    p.add_argument("--out", default="checkpoints_pretrain", type=Path)
    # Architecture (must match supervised encoder for warm-start to work).
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--ffn", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=256)
    # MLM
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=96)
    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    return p


def pretrain_main(argv: List[str] | None = None) -> int:
    args = build_pretrain_parser().parse_args(argv)
    from khmer_g2p.neural.pretrain import PretrainConfig, pretrain

    cfg = PretrainConfig(
        corpus_path=str(args.corpus),
        lexicon_path=str(args.lexicon) if args.lexicon else None,
        out_dir=str(args.out),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_len=args.max_len,
        mlm_prob=args.mlm_prob,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
    )
    best = pretrain(cfg)
    print(f"\nPretrained encoder -> {best}")
    return 0


if __name__ == "__main__":
    # Allow `python -m khmer_g2p.neural.cli {train,predict,finetune,pretrain} ...`
    sub_map = {
        "train": train_main, "predict": predict_main,
        "finetune": finetune_main, "pretrain": pretrain_main,
    }
    if len(sys.argv) > 1 and sys.argv[1] in sub_map:
        sub = sys.argv.pop(1)
        raise SystemExit(sub_map[sub]())
    raise SystemExit(train_main())
