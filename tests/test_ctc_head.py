"""Smoke tests for the optional CTC auxiliary head on the encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.model import G2PConfig, G2PTransformer


def _make_model(use_ctc: bool, src_v: int = 20, tgt_v: int = 30) -> G2PTransformer:
    cfg = G2PConfig(
        src_vocab_size=src_v,
        tgt_vocab_size=tgt_v,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_len=32,
        pad_id=0,
        use_ctc=use_ctc,
    )
    return G2PTransformer(cfg)


def test_ctc_head_off_by_default():
    model = _make_model(use_ctc=False)
    assert model.ctc_proj is None


def test_ctc_head_on_when_enabled():
    model = _make_model(use_ctc=True)
    assert model.ctc_proj is not None
    assert isinstance(model.ctc_proj, torch.nn.Linear)
    assert model.ctc_proj.out_features == 30  # tgt_vocab_size


def test_ctc_log_probs_shape():
    model = _make_model(use_ctc=True)
    B, T = 3, 7
    src = torch.randint(1, 20, (B, T))
    memory, _ = model.encode(src)
    log_probs = model.ctc_log_probs(memory)
    # nn.functional.ctc_loss wants (T, B, V)
    assert log_probs.shape == (T, B, 30)
    # log-softmax rows sum to ~1 (in probability space)
    probs = log_probs.exp()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_ctc_log_probs_errors_when_disabled():
    model = _make_model(use_ctc=False)
    src = torch.randint(1, 20, (2, 5))
    memory, _ = model.encode(src)
    with pytest.raises(RuntimeError, match="CTC head not enabled"):
        model.ctc_log_probs(memory)


def test_joint_ce_plus_ctc_loss_backprop():
    """End-to-end: build CE+CTC loss and confirm gradients flow into both heads."""
    torch.manual_seed(0)
    model = _make_model(use_ctc=True)
    B, T_src, T_tgt = 4, 10, 6
    src = torch.randint(1, 20, (B, T_src))
    tgt_in = torch.randint(4, 30, (B, T_tgt))
    tgt_out = torch.randint(4, 30, (B, T_tgt))

    # Decoder CE branch
    memory, src_kpm = model.encode(src)
    logits = model.decode(tgt_in, memory, src_kpm)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
    )

    # Encoder CTC branch
    ctc_log_probs = model.ctc_log_probs(memory)  # (T_src, B, V)
    input_lens = torch.full((B,), T_src, dtype=torch.long)
    target_lens = torch.full((B,), T_tgt, dtype=torch.long)
    flat_targets = tgt_out.reshape(-1)
    ctc = torch.nn.functional.ctc_loss(
        ctc_log_probs, flat_targets, input_lens, target_lens,
        blank=0, reduction="mean", zero_infinity=True,
    )

    loss = ce + 0.3 * ctc
    loss.backward()

    # Gradients should flow into both output heads.
    assert model.out_proj.weight.grad is not None
    assert model.ctc_proj.weight.grad is not None
    assert torch.isfinite(loss).item()


def test_config_roundtrip_preserves_use_ctc():
    cfg = G2PConfig(src_vocab_size=10, tgt_vocab_size=10, use_ctc=True)
    d = cfg.to_dict()
    assert d["use_ctc"] is True
    cfg2 = G2PConfig.from_dict(d)
    assert cfg2.use_ctc is True


def test_config_backcompat_without_use_ctc_key():
    """Old checkpoints (pre-0.3) have no 'use_ctc' key — should default to False."""
    old_payload = {
        "src_vocab_size": 10,
        "tgt_vocab_size": 10,
        "d_model": 192,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "max_len": 256,
        "pad_id": 0,
    }
    cfg = G2PConfig.from_dict(old_payload)
    assert cfg.use_ctc is False
