"""Tests for the per-position consonant-series and syllable-boundary labels."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from khmer_g2p.neural.aux_labels import (
    IGNORE_INDEX,
    extract_aux_labels,
    extract_series_labels,
    extract_syllable_labels,
)
from khmer_g2p.neural.vocab import TOKENIZER_CHAR, Vocab


def _vocab_with(chars: str) -> Vocab:
    return Vocab.from_tokens(list(chars), tokenizer=TOKENIZER_CHAR)


def _encode_padded(words, vocab) -> torch.Tensor:
    """Encode each word with EOS, then right-pad to a square batch tensor."""
    rows = [vocab.encode(w, add_eos=True) for w in words]
    T = max(len(r) for r in rows)
    pad = vocab.pad_id
    padded = [r + [pad] * (T - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# series labels
# ---------------------------------------------------------------------------

def test_series_labels_pick_out_consonants():
    # ក = series 1, គ = series 2, ា = vowel sign (ignored).
    vocab = _vocab_with("កគា")
    src = _encode_padded(["កគា", "ក"], vocab)
    lbl = extract_series_labels(src, vocab)
    assert lbl.shape == src.shape
    # Row 0: ក (S1) → 0 ; គ (S2) → 1 ; ា (vowel) → -100 ; EOS → -100
    assert lbl[0, 0].item() == 0
    assert lbl[0, 1].item() == 1
    assert lbl[0, 2].item() == IGNORE_INDEX
    assert lbl[0, 3].item() == IGNORE_INDEX                    # EOS
    # Row 1: ក (S1) → 0 ; EOS → -100 ; PAD → -100
    assert lbl[1, 0].item() == 0
    assert lbl[1, 1].item() == IGNORE_INDEX
    assert lbl[1, 2].item() == IGNORE_INDEX


def test_series_labels_ignore_pad_and_specials():
    vocab = _vocab_with("ក")
    src = _encode_padded(["ក"], vocab)                          # [ID(ក), EOS]
    lbl = extract_series_labels(src, vocab)
    # Only the very first position should be a real label.
    assert (lbl == IGNORE_INDEX).sum().item() == src.numel() - 1


# ---------------------------------------------------------------------------
# syllable boundary labels
# ---------------------------------------------------------------------------

def test_syllable_boundary_marks_first_char_of_each_syllable():
    # "ខ្មែរ" parses as a single syllable with 4 chars: ខ ្ ម ែ … plus a
    # final "រ" inside the same syllable in many analyses, so we instead use
    # a clearly multi-syllabic word. Two consonants joined by neither COENG
    # nor a vowel will syllabify into two separate syllables.
    vocab = _vocab_with("កខ")
    src = _encode_padded(["កខ"], vocab)
    lbl = extract_syllable_labels(src, vocab)
    assert lbl.shape == src.shape
    # Position 0 (ក) starts syllable 0 → label 1.
    # Position 1 (ខ) starts syllable 1 → label 1.
    assert lbl[0, 0].item() == 1
    assert lbl[0, 1].item() == 1
    # Trailing EOS / PAD are ignored.
    assert lbl[0, 2].item() == IGNORE_INDEX


def test_syllable_boundary_continuations_are_zero():
    # ក + ា (vowel sign) stays one syllable: 2 chars, labels [1, 0].
    vocab = _vocab_with("កា")
    src = _encode_padded(["កា"], vocab)
    lbl = extract_syllable_labels(src, vocab)
    assert lbl[0, 0].item() == 1                                # syllable start
    assert lbl[0, 1].item() == 0                                # continuation
    assert lbl[0, 2].item() == IGNORE_INDEX                     # EOS


# ---------------------------------------------------------------------------
# combined extractor
# ---------------------------------------------------------------------------

def test_extract_aux_labels_returns_both():
    vocab = _vocab_with("កគ")
    src = _encode_padded(["កគ"], vocab)
    series, syll = extract_aux_labels(src, vocab)
    # Series labels for two distinct consonants:
    assert series[0, 0].item() == 0                             # ក = S1
    assert series[0, 1].item() == 1                             # គ = S2
    # Both consonants start their own syllables:
    assert syll[0, 0].item() == 1
    assert syll[0, 1].item() == 1
