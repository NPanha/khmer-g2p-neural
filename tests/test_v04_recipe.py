"""Tests for the v0.4 training recipe.

Covers:
    * Tied input/output embeddings — actually share storage and survive save/load.
    * Cosine LR schedule shape (warmup → cosine → min floor).
    * Weight EMA: shadow updates, apply_to context manager swaps and restores.
    * End-to-end: a minimal `train(...)` run with the v0.4 stack writes a
      best.pt that loads cleanly through `NeuralG2P.from_checkpoint`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.ema import ModelEma
from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.train import (
    TrainConfig,
    _cosine_lr,
    _resolve_amp,
    train,
)


# ---------------------------------------------------------------------------
# Tied embeddings
# ---------------------------------------------------------------------------

def test_tie_embeddings_shares_storage():
    cfg = G2PConfig(
        src_vocab_size=12, tgt_vocab_size=12,
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_len=8,
        tie_embeddings=True,
    )
    model = G2PTransformer(cfg)
    assert model.out_proj.weight is model.tgt_embed.weight
    # Mutating one mutates the other.
    with torch.no_grad():
        model.tgt_embed.weight[0, 0] = 1.2345
    assert torch.allclose(
        model.out_proj.weight[0, 0], torch.tensor(1.2345)
    )


def test_tie_embeddings_off_by_default_for_backcompat():
    cfg = G2PConfig(src_vocab_size=10, tgt_vocab_size=10)
    assert cfg.tie_embeddings is False
    model = G2PTransformer(cfg)
    assert model.out_proj.weight is not model.tgt_embed.weight


def test_tie_after_load_state_dict(tmp_path):
    """load_state_dict copies into each parameter independently and silently
    breaks tying. ``tie_weights_if_configured`` must restore it."""
    cfg = G2PConfig(
        src_vocab_size=10, tgt_vocab_size=10,
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_len=8,
        tie_embeddings=True,
    )
    src = G2PTransformer(cfg)
    p = tmp_path / "tied.pt"
    torch.save(src.state_dict(), p)

    dst = G2PTransformer(cfg)
    dst.load_state_dict(torch.load(p, weights_only=False))
    # Right after raw load_state_dict, storage may have been unshared.
    dst.tie_weights_if_configured()
    assert dst.out_proj.weight is dst.tgt_embed.weight


# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------

def test_cosine_lr_warmup_endpoints():
    peak, warmup, total = 5e-4, 100, 1000
    # Step 1 of warmup is roughly 1/100 of peak.
    assert _cosine_lr(1, peak, warmup, total) == pytest.approx(peak / warmup, rel=1e-6)
    # End of warmup is exactly the peak.
    assert _cosine_lr(warmup, peak, warmup, total) == pytest.approx(peak)


def test_cosine_lr_decays_to_min_ratio():
    peak, warmup, total = 5e-4, 100, 1000
    # At step==total, cosine(π) = -1 → floor = peak * min_ratio.
    floor = _cosine_lr(total, peak, warmup, total, min_ratio=0.05)
    assert floor == pytest.approx(peak * 0.05, rel=1e-6)


def test_cosine_lr_is_monotone_decreasing_after_warmup():
    peak, warmup, total = 5e-4, 100, 1000
    last = _cosine_lr(warmup, peak, warmup, total)
    for s in range(warmup + 1, total + 1, 50):
        cur = _cosine_lr(s, peak, warmup, total)
        assert cur <= last + 1e-12, f"non-monotone at step {s}: {cur} > {last}"
        last = cur


# ---------------------------------------------------------------------------
# AMP resolver
# ---------------------------------------------------------------------------

def test_amp_off_on_cpu():
    enabled, dtype = _resolve_amp("auto", torch.device("cpu"))
    assert enabled is False
    assert dtype is None


def test_amp_explicit_off():
    enabled, _ = _resolve_amp("off", torch.device("cuda" if torch.cuda.is_available()
                                                  else "cpu"))
    assert enabled is False


# ---------------------------------------------------------------------------
# ModelEma
# ---------------------------------------------------------------------------

def test_ema_shadow_initially_matches_model():
    cfg = G2PConfig(
        src_vocab_size=10, tgt_vocab_size=10,
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_len=8,
    )
    model = G2PTransformer(cfg)
    ema = ModelEma(model, decay=0.9)
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            assert torch.allclose(ema.state_dict()[k], v.float())


def test_ema_update_pulls_toward_live_weights():
    cfg = G2PConfig(
        src_vocab_size=10, tgt_vocab_size=10,
        d_model=8, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, dropout=0.0, max_len=4,
    )
    model = G2PTransformer(cfg)
    ema = ModelEma(model, decay=0.5, use_warmup=False)

    # Snapshot baseline.
    baseline = {k: v.clone() for k, v in ema.state_dict().items()}

    # Perturb the live model and update the EMA once.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))
    ema.update(model, step=1000)  # well past warmup; effective decay = 0.5

    # Each EMA parameter should now be ~ 0.5*old + 0.5*(old + 1) = old + 0.5.
    sd = ema.state_dict()
    sample_key = next(k for k, v in sd.items()
                      if v.is_floating_point() and v.numel() > 0)
    diff = (sd[sample_key] - baseline[sample_key]).abs().mean().item()
    assert 0.3 < diff < 0.7, f"expected ~0.5 mean shift; got {diff}"


def test_ema_apply_to_swaps_and_restores():
    cfg = G2PConfig(
        src_vocab_size=10, tgt_vocab_size=10,
        d_model=8, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, dropout=0.0, max_len=4,
    )
    model = G2PTransformer(cfg)
    ema = ModelEma(model, decay=0.5, use_warmup=False)

    # Diverge the live and EMA weights.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 5.0)

    live_sample = next(iter(model.parameters())).detach().clone()
    ema_sample = ema.state_dict()[next(iter(model.state_dict()))]

    with ema.apply_to(model):
        # Inside the context, the model carries the EMA weights.
        first_param = next(iter(model.parameters())).detach()
        # The EMA wasn't updated yet → still equals the original (pre-perturb) weights.
        assert not torch.allclose(first_param, live_sample)

    # After the context, live weights are back.
    assert torch.allclose(next(iter(model.parameters())).detach(), live_sample)


# ---------------------------------------------------------------------------
# End-to-end smoke: v0.4 stack writes a checkpoint that NeuralG2P can load
# ---------------------------------------------------------------------------

def _write_tiny_tsv(path: Path) -> None:
    rows = [
        ("ka", "k a"), ("ma", "m a"), ("ki", "k i"), ("nu", "n u"),
        ("ko", "k o"), ("mo", "m o"), ("ku", "k u"), ("ni", "n i"),
    ] * 4
    path.write_text("\n".join(f"{w}\t{p}" for w, p in rows), encoding="utf-8")


def test_v04_train_and_load_e2e(tmp_path):
    data = tmp_path / "lex.tsv"
    _write_tiny_tsv(data)
    out = tmp_path / "v04"

    cfg = TrainConfig(
        data_path=str(data),
        out_dir=str(out),
        d_model=16, nhead=2,
        num_encoder_layers=2, num_decoder_layers=1,   # asymmetric
        dim_feedforward=32, dropout=0.1, max_len=8,
        tie_embeddings=True,
        batch_size=4, epochs=2,
        lr=1e-3, warmup_steps=2,
        weight_decay=0.0, label_smoothing=0.0, grad_clip=1.0,
        lr_schedule="cosine", min_lr_ratio=0.1,
        ema_decay=0.9,
        amp="off",
        val_frac=0.25, test_frac=0.25, seed=0,
        device="cpu", num_workers=0, log_every=10,
        patience=3, tgt_tokenizer="char",
        progress=False,
        use_ctc=False, ctc_weight=0.0,
    )
    best = train(cfg)
    assert best.exists()

    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["config"]["tie_embeddings"] is True
    assert payload.get("from_ema") is True

    # Round-trip: load through NeuralG2P (this exercises tie_weights_if_configured).
    g2p = NeuralG2P.from_checkpoint(best, device="cpu")
    out_str = g2p.convert("ka")
    assert isinstance(out_str, str)
    # Tying must survive the load.
    assert g2p.model.out_proj.weight is g2p.model.tgt_embed.weight


def test_v04_history_records_lr_per_epoch(tmp_path):
    data = tmp_path / "lex.tsv"
    _write_tiny_tsv(data)
    out = tmp_path / "v04hist"

    cfg = TrainConfig(
        data_path=str(data), out_dir=str(out),
        d_model=16, nhead=2,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_len=8,
        tie_embeddings=False,
        batch_size=4, epochs=2,
        lr=1e-3, warmup_steps=2,
        lr_schedule="cosine", min_lr_ratio=0.1,
        ema_decay=0.0, amp="off",
        val_frac=0.25, test_frac=0.25, seed=0,
        device="cpu", num_workers=0,
        progress=False,
        use_ctc=False,
    )
    train(cfg)
    history = json.loads((out / "history.json").read_text())
    assert all("lr" in h for h in history)
    # LR should be on the cosine curve — strictly positive.
    assert all(0.0 < h["lr"] <= 1e-3 + 1e-9 for h in history)
