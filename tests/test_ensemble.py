"""Tests for EnsembleG2P (multi-seed log-prob averaging)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.ensemble import EnsembleG2P
from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.vocab import TOKENIZER_CHAR, Vocab


def _make_g2p(seed: int) -> NeuralG2P:
    torch.manual_seed(seed)
    src = Vocab.from_tokens(list("kxmnpaeiou"), tokenizer=TOKENIZER_CHAR)
    tgt = Vocab.from_tokens(list("kʰmnaiu"),     tokenizer=TOKENIZER_CHAR)
    cfg = G2PConfig(
        src_vocab_size=len(src), tgt_vocab_size=len(tgt),
        d_model=32, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=64, dropout=0.0, max_len=16,
        pad_id=src.pad_id,
    )
    return NeuralG2P(G2PTransformer(cfg), src, tgt, max_len=8)


def test_ensemble_requires_multiple_members():
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleG2P([_make_g2p(0)])


def test_ensemble_rejects_mismatched_vocab():
    a = _make_g2p(1)
    # Build a second model with a *different* tgt vocab.
    src = a.src_vocab
    tgt_other = Vocab.from_tokens(list("xyz"), tokenizer=TOKENIZER_CHAR)
    cfg = G2PConfig(
        src_vocab_size=len(src), tgt_vocab_size=len(tgt_other),
        d_model=32, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=64, dropout=0.0, max_len=16,
        pad_id=src.pad_id,
    )
    b = NeuralG2P(G2PTransformer(cfg), src, tgt_other, max_len=8)
    with pytest.raises(ValueError, match="tgt_vocab differs"):
        EnsembleG2P([a, b])


def test_ensemble_greedy_returns_string():
    e = EnsembleG2P([_make_g2p(1), _make_g2p(2), _make_g2p(3)])
    out = e.convert("kma", beam=1)
    assert isinstance(out, str)


def test_ensemble_beam_returns_string():
    e = EnsembleG2P([_make_g2p(11), _make_g2p(22)])
    out = e.convert("kma", beam=4, length_penalty=0.6)
    assert isinstance(out, str)


def test_ensemble_step_log_probs_average_matches_manual_mean():
    """Average of log-softmax across members should equal what the helper produces."""
    e = EnsembleG2P([_make_g2p(101), _make_g2p(202)])
    word = "kma"
    memos = e._encode_all(word)
    bos = e.tgt_vocab.bos_id
    ys = torch.tensor([[bos]], dtype=torch.long, device=e.device)

    # Manual: log-softmax each member separately, then average.
    manual = None
    for member, (mem, kpm) in zip(e.members, memos):
        logits = member.model.decode(ys.to(member.device), mem, kpm)[:, -1]
        lp = torch.log_softmax(logits, dim=-1)[0].to(e.device)
        manual = lp if manual is None else manual + lp
    manual = manual / len(e.members)

    helper = e._step_log_probs(ys, memos)
    assert torch.allclose(helper, manual, atol=1e-6)


def test_ensemble_convert_batch_signature_matches_neural():
    e = EnsembleG2P([_make_g2p(7), _make_g2p(8)])
    outs = e.convert_batch(["ka", "mi"], beam=2, length_penalty=0.6)
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)
