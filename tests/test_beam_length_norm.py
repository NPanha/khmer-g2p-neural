"""Tests for the GNMT-style length penalty in NeuralG2P beam search."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.vocab import (
    BOS, EOS, PAD, UNK, TOKENIZER_CHAR, Vocab,
)


def _toy_g2p() -> NeuralG2P:
    """Tiny untrained model — just enough to exercise the decoding code."""
    src_v = Vocab.from_tokens(list("kxmnpaeiou"), tokenizer=TOKENIZER_CHAR)
    tgt_v = Vocab.from_tokens(list("kʰmnaiu"),     tokenizer=TOKENIZER_CHAR)
    cfg = G2PConfig(
        src_vocab_size=len(src_v),
        tgt_vocab_size=len(tgt_v),
        d_model=32, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=64, dropout=0.0, max_len=16,
        pad_id=src_v.pad_id,
    )
    model = G2PTransformer(cfg)
    return NeuralG2P(model, src_v, tgt_v, max_len=12)


def test_length_norm_divisor_monotone():
    """Larger lengths → larger divisor → discount on raw cumulative log-prob."""
    f = NeuralG2P._length_norm
    assert f(1, 0.6) < f(5, 0.6) < f(20, 0.6)
    # alpha = 0 disables the penalty.
    assert f(1, 0.0) == f(50, 0.0) == 1.0


def test_length_norm_returns_string():
    """Beam decoding still returns a string regardless of length penalty."""
    g = _toy_g2p()
    out_default = g.convert("kmae", beam=4)
    out_no_pen = g.convert("kmae", beam=4, length_penalty=0.0)
    assert isinstance(out_default, str)
    assert isinstance(out_no_pen, str)


def test_length_penalty_changes_beam_ranking_in_constructed_case():
    """Build two finished beams by hand and confirm the GNMT divisor reorders them."""
    # Two beams: short (raw=-1.5, len=2) and long (raw=-3.0, len=8).
    # With alpha=0, raw log-prob wins → short.
    # With alpha=0.6, long gets a smaller divisor *relative to its raw advantage*…
    # actually the divisor is bigger for longer → makes the long score *worse*.
    # So the test asserts the ordering CAN flip when we make the long beam's
    # raw log-prob just barely worse, by tuning alpha.
    a = (-1.0, 3)   # raw, length
    b = (-1.3, 8)   # raw, length
    f = NeuralG2P._length_norm

    # With alpha=0: pure raw log-prob → a wins.
    assert (a[0] / f(a[1], 0.0)) > (b[0] / f(b[1], 0.0))

    # With alpha=0.6: longer 'b' gets divided by a *bigger* divisor; since the
    # raws are negative, dividing by >1 makes the score *less negative*, i.e.
    # better. So 'b' becomes preferable.
    assert (b[0] / f(b[1], 0.6)) > (a[0] / f(a[1], 0.6))


def test_convert_batch_passes_length_penalty_through():
    g = _toy_g2p()
    outs = g.convert_batch(["kma", "mei"], beam=2, length_penalty=0.6)
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)
