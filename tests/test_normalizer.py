"""Tests for the Khmer normalizer / syllabifier."""

from khmer_g2p.normalizer import normalize, syllabify, tokenize


def test_normalize_strips_zwsp():
    s = "ខ្ញុំ\u200Bអ្នក"
    assert "\u200B" not in normalize(s)


def test_normalize_khmer_digits():
    assert normalize("១២៣") == "123"


def test_syllabify_simple_initial_only():
    syls = syllabify("ក")
    assert len(syls) == 1
    assert syls[0].initial == "ក"
    assert syls[0].vowel == ""
    assert syls[0].final == ""


def test_syllabify_with_subscript_and_vowel():
    # ខ្ញុំ = ខ + COENG + ញ + ុ + ំ
    syls = syllabify("ខ្ញុំ")
    assert len(syls) == 1
    s = syls[0]
    assert s.initial == "ខ"
    assert s.subscripts == ["ញ"]
    assert s.vowel == "\u17BB"             # ុ is the nuclear vowel
    assert s.vowel_final_sign == "\u17C6"  # ំ is the trailing nasal coda


def test_tokenize_mixed():
    tokens = tokenize("Hello ខ្មែរ 2026")
    assert any("\u1780" <= t[0] <= "\u17FF" for t in tokens)
    assert any(t.strip().isascii() for t in tokens)
