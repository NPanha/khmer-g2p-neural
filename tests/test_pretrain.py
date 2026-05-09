"""Smoke tests for masked-character encoder pretraining + warm-start."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.pretrain import PretrainConfig, pretrain
from khmer_g2p.neural.train import TrainConfig, train, _load_pretrained_encoder
from khmer_g2p.neural.model import G2PConfig, G2PTransformer


def _write_corpus(path: Path) -> None:
    # Tiny synthetic Khmer-ish text — repeated so a few windows fit.
    lines = ["ខ្មែរភាសាសាលាកុំព្យូទ័រ" * 8] * 16
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_lex(path: Path) -> None:
    rows = [("ខ្មែរ", "k m"), ("ភាសា", "p a"), ("សាលា", "s a")] * 4
    path.write_text("\n".join(f"{w}\t{ipa}" for w, ipa in rows), encoding="utf-8")


def test_pretrain_smoke_writes_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt";  _write_corpus(corpus)
    lex    = tmp_path / "lex.tsv";     _write_lex(lex)
    out    = tmp_path / "pre"

    cfg = PretrainConfig(
        corpus_path=str(corpus), lexicon_path=str(lex), out_dir=str(out),
        d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
        dropout=0.1, max_len=64,
        mlm_prob=0.5, window=16, stride=8, min_chars_per_window=4,
        batch_size=4, epochs=2, lr=1e-3, warmup_steps=2, min_lr_ratio=0.1,
        weight_decay=0.0, grad_clip=1.0,
        amp="off", device="cpu", seed=0, num_workers=0, progress=False,
    )
    best = pretrain(cfg)
    assert best.exists()
    assert (out / "history.json").exists()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    # Required keys for the supervised loader to consume the pretrain ckpt.
    assert "src_vocab" in payload
    assert "model_state" in payload
    keys = list(payload["model_state"].keys())
    assert any(k.startswith("src_embed.")           for k in keys)
    assert any(k.startswith("transformer.encoder.") for k in keys)


def test_load_pretrained_encoder_copies_only_encoder(tmp_path):
    """Make sure the warm-start helper transfers exactly the encoder."""
    # Build a supervised model with a vocab compatible with the pretrain.
    src_vocab_size, tgt_vocab_size = 8, 6
    cfg = G2PConfig(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_len=8,
    )
    model = G2PTransformer(cfg)

    # Build a fake "pretrain" state: random tensors that match the encoder shapes.
    pre_state = {}
    for k, v in model.state_dict().items():
        pre_state[k] = torch.randn_like(v) if v.is_floating_point() else v.clone()

    # Snapshot decoder before — it should NOT change.
    dec_before = {k: v.detach().clone()
                  for k, v in model.state_dict().items()
                  if k.startswith("transformer.decoder.")}

    n = _load_pretrained_encoder(model, pre_state)
    assert n > 0

    # Encoder weights now equal the pretrain ones.
    for k in model.state_dict():
        if k.startswith("transformer.encoder.") or k.startswith("src_embed.") \
                or k.startswith("src_pos."):
            assert torch.allclose(model.state_dict()[k], pre_state[k])

    # Decoder weights are untouched.
    for k, v in dec_before.items():
        assert torch.allclose(model.state_dict()[k], v)


def test_supervised_train_consumes_pretrained_encoder(tmp_path):
    """End-to-end: pretrain → supervised train with warm-start."""
    corpus = tmp_path / "corpus.txt";  _write_corpus(corpus)
    lex    = tmp_path / "lex.tsv";     _write_lex(lex)

    pre_out = tmp_path / "pre"
    pre_cfg = PretrainConfig(
        corpus_path=str(corpus), lexicon_path=str(lex), out_dir=str(pre_out),
        d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
        dropout=0.1, max_len=64,
        mlm_prob=0.5, window=16, stride=8, min_chars_per_window=4,
        batch_size=4, epochs=1, lr=1e-3, warmup_steps=2,
        amp="off", device="cpu", seed=0, num_workers=0, progress=False,
    )
    pre_ckpt = pretrain(pre_cfg)

    sup_out = tmp_path / "sup"
    sup_cfg = TrainConfig(
        data_path=str(lex), out_dir=str(sup_out),
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.1, max_len=64,
        tie_embeddings=False,
        batch_size=4, epochs=1, lr=1e-3, warmup_steps=2,
        lr_schedule="cosine", min_lr_ratio=0.1,
        ema_decay=0.0, amp="off",
        val_frac=0.25, test_frac=0.25, seed=0,
        device="cpu", num_workers=0, progress=False,
        use_ctc=False, ctc_weight=0.0,
        pretrained_encoder=str(pre_ckpt),
    )
    best = train(sup_cfg)
    assert best.exists()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    # Source vocab should match the pretrain's vocab (warm-start path).
    pre_vocab = torch.load(pre_ckpt, map_location="cpu",
                           weights_only=False)["src_vocab"]
    assert payload["src_vocab"]["itos"] == pre_vocab["itos"]
