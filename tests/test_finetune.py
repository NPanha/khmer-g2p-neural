"""Smoke tests for the low-LR resume fine-tune entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.finetune import FinetuneConfig, _set_dropout, finetune
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.train import TrainConfig, train


# ---------------------------------------------------------------------------
# _set_dropout
# ---------------------------------------------------------------------------

def test_set_dropout_walks_the_full_module_tree():
    cfg = G2PConfig(
        src_vocab_size=10, tgt_vocab_size=10,
        d_model=16, nhead=2,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=32, dropout=0.5, max_len=8,
    )
    model = G2PTransformer(cfg)

    # Sanity: at least one Dropout submodule exists with p=0.5.
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops, "expected nn.Dropout submodules in the Transformer stack"
    assert all(d.p == pytest.approx(0.5) for d in drops)

    n = _set_dropout(model, 0.1)
    assert n == len(drops)
    assert all(d.p == pytest.approx(0.1) for d in drops)


# ---------------------------------------------------------------------------
# end-to-end finetune on a synthetic 4-pair "lexicon" — pure smoke
# ---------------------------------------------------------------------------

def _write_tiny_tsv(path: Path) -> None:
    rows = [
        ("ka",  "k a"),
        ("ma",  "m a"),
        ("ki",  "k i"),
        ("nu",  "n u"),
    ] * 4  # 16 pairs total — split allows tiny train/val/test
    path.write_text("\n".join(f"{w}\t{p}" for w, p in rows), encoding="utf-8")


def test_finetune_smoke_runs_and_writes_checkpoint(tmp_path):
    data = tmp_path / "lex.tsv"
    _write_tiny_tsv(data)

    train_dir = tmp_path / "train"
    train_cfg = TrainConfig(
        data_path=str(data),
        out_dir=str(train_dir),
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.3, max_len=8,
        batch_size=4, epochs=2, lr=1e-3, warmup_steps=4,
        weight_decay=0.0, label_smoothing=0.0, grad_clip=1.0,
        val_frac=0.25, test_frac=0.25, seed=0,
        device="cpu", num_workers=0, log_every=10,
        patience=3, tgt_tokenizer="char",
        progress=False,
        use_ctc=False, ctc_weight=0.0,
    )
    best_train = train(train_cfg)
    assert best_train.exists(), "main training must produce best.pt"

    ft_dir = tmp_path / "ft"
    ft_cfg = FinetuneConfig(
        ckpt_path=str(best_train),
        data_path=str(data),
        out_dir=str(ft_dir),
        lr=1e-4, dropout=0.1,
        epochs=2, patience=2,
        batch_size=4,
        weight_decay=0.0, label_smoothing=0.0,
        ctc_weight=0.0,
        val_frac=0.25, test_frac=0.25, seed=0,
        device="cpu", num_workers=0,
        progress=False,
    )
    best_ft = finetune(ft_cfg)

    assert best_ft.exists(), "finetune must write a best.pt"
    payload = torch.load(best_ft, map_location="cpu", weights_only=False)
    assert payload.get("finetune") is True
    assert payload.get("finetune_from") == str(best_train)
    assert (ft_dir / "history.json").exists()
    history = json.loads((ft_dir / "history.json").read_text())
    assert len(history) >= 1
    assert "val_per" in history[0]
