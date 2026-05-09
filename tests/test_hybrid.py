"""Tests for HybridG2P (lexicon-first, neural-fallback)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.hybrid import HybridG2P
from khmer_g2p.lexicon import Lexicon
from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.vocab import TOKENIZER_CHAR, Vocab


class _StubNeural:
    """NeuralG2P stand-in that records every call so we can assert routing."""

    def __init__(self) -> None:
        self.calls: list = []

    def convert(self, word: str, beam: int = 1, length_penalty: float = 0.6) -> str:
        self.calls.append((word, beam, length_penalty))
        return f"<model:{word}>"


def test_lexicon_hit_returns_canonical_ipa_and_skips_model():
    stub = _StubNeural()
    lex = Lexicon([("ខ្មែរ", "kʰmae"), ("ភាសា", "piəsaː")])
    hyb = HybridG2P(stub, lex)  # type: ignore[arg-type]

    assert hyb.convert("ខ្មែរ") == "kʰmae"
    assert hyb.last_source == "lexicon"
    assert stub.calls == [], "model should not be called on a lexicon hit"


def test_lexicon_miss_falls_through_to_model():
    stub = _StubNeural()
    lex = Lexicon([("ខ្មែរ", "kʰmae")])
    hyb = HybridG2P(stub, lex)  # type: ignore[arg-type]

    out = hyb.convert("ទេស", beam=4, length_penalty=0.5)
    assert out == "<model:ទេស>"
    assert hyb.last_source == "model"
    assert stub.calls == [("ទេស", 4, 0.5)], \
        "miss must forward beam and length_penalty to the model"


def test_dict_constructor_normalizes_keys():
    stub = _StubNeural()
    hyb = HybridG2P(stub, {"ខ្មែរ": "kʰmae"})  # type: ignore[arg-type]
    assert "ខ្មែរ" in hyb
    assert hyb.convert("ខ្មែរ") == "kʰmae"


def test_hit_rate_and_counters():
    stub = _StubNeural()
    lex = Lexicon([("ក", "k"), ("ខ", "kʰ")])
    hyb = HybridG2P(stub, lex)  # type: ignore[arg-type]

    hyb.convert("ក")          # hit
    hyb.convert("ខ")          # hit
    hyb.convert("UNKNOWN")     # miss
    assert hyb.hits == 2
    assert hyb.misses == 1
    assert abs(hyb.hit_rate - 2 / 3) < 1e-9

    hyb.reset_stats()
    assert hyb.hits == 0 and hyb.misses == 0
    assert hyb.hit_rate == 0.0


def test_convert_batch_returns_per_word_results():
    stub = _StubNeural()
    lex = Lexicon([("ក", "k")])
    hyb = HybridG2P(stub, lex)  # type: ignore[arg-type]

    outs = hyb.convert_batch(["ក", "x"])
    assert outs == ["k", "<model:x>"]
    assert hyb.hits == 1 and hyb.misses == 1


def test_with_real_neural_model_smoke():
    """End-to-end smoke: real (untrained) NeuralG2P + small Lexicon."""
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
    neural = NeuralG2P(model, src_v, tgt_v, max_len=10)
    lex = Lexicon([("kma", "kʰma")])
    hyb = HybridG2P(neural, lex)

    # Hit returns canonical IPA verbatim — no model involved.
    assert hyb.convert("kma") == "kʰma"
    # Miss returns a string (whatever the untrained model produced).
    out = hyb.convert("xnu", beam=2)
    assert isinstance(out, str)
