"""Tests for the Khmer IPA phoneme tokenizer and phoneme-mode Vocab."""

import pytest

# These tests only need pure Python — no torch required.
from khmer_g2p.neural.phoneme_tokenizer import (
    tokenize,
    detokenize,
    flag_unknown,
    MODIFIERS,
)
from khmer_g2p.neural.vocab import Vocab, build_vocabs, TOKENIZER_PHONEME


# --- Tokenizer -------------------------------------------------------------


def test_length_mark_glues_to_vowel():
    assert tokenize("kɑː") == ["k", "ɑː"]


def test_aspiration_glues_to_consonant():
    assert tokenize("pʰɔː") == ["pʰ", "ɔː"]


def test_stress_and_syllable_separator_are_standalone():
    assert tokenize("kɑ.ˈkaːj") == ["k", "ɑ", ".", "ˈ", "k", "aː", "j"]


def test_diphthong_stays_as_two_tokens():
    # 'iə' is two vowel tokens by design — keeps the vocab small and lets
    # the model handle Khmer's many vowel-vowel sequences compositionally.
    assert tokenize("niːə") == ["n", "iː", "ə"]


def test_breve_is_its_own_token():
    # 'ĕ' is a precomposed Latin letter with breve — a single codepoint.
    assert tokenize("jĕəʔ") == ["j", "ĕ", "ə", "ʔ"]


def test_word_separator_is_preserved():
    assert tokenize("kɑː cɔː") == ["k", "ɑː", " ", "c", "ɔː"]


def test_roundtrip_is_lossless():
    samples = [
        "pʰɔː",
        "ˈkeː",
        "jĕəʔ",
        "kɑː cɔː pʰɔː kʰɑː",
        ".ʔuː.paʔ.niː.jĕəʔ.kam",
        "kɑ.ˈkaːj",
        "ʋi.ˈseh",
        "neək",
    ]
    for s in samples:
        assert detokenize(tokenize(s)) == s


def test_stray_modifier_is_preserved_not_dropped():
    # A length mark at the start has nothing to attach to — keep it
    # as its own token rather than silently drop it, so callers can
    # detect malformed input.
    assert tokenize("ːa") == ["ː", "a"]


def test_modifiers_set_is_length_and_aspiration():
    assert MODIFIERS == {"ː", "ʰ"}


def test_flag_unknown_catches_garbage():
    toks = tokenize("kɑː")
    assert flag_unknown(toks) == []
    # Pure Latin-script 'x' is not in the Khmer IPA inventory.
    assert flag_unknown(["x", "k"]) == ["x"]


# --- Vocab integration -----------------------------------------------------


def test_phoneme_vocab_has_multi_char_tokens():
    pairs = [("ក", "kɑː"), ("ខ", "kʰɑː"), ("គា", "kiə")]
    _, tgt = build_vocabs(pairs, tgt_tokenizer=TOKENIZER_PHONEME)
    # 'kʰ' and 'ɑː' should be single tokens in phoneme mode.
    assert "kʰ" in tgt.stoi
    assert "ɑː" in tgt.stoi
    # Length / aspiration marks should NOT appear as standalone tokens —
    # they're always glued.
    assert "ː" not in tgt.stoi
    assert "ʰ" not in tgt.stoi


def test_phoneme_vocab_roundtrip():
    pairs = [("ក", "kɑː"), ("ខ", "pʰɔː"), ("គា", "niː")]
    _, tgt = build_vocabs(pairs, tgt_tokenizer=TOKENIZER_PHONEME)
    for _, ipa in pairs:
        ids = tgt.encode(ipa)
        assert tgt.decode(ids) == ipa


def test_phoneme_encoding_is_shorter_than_char():
    pairs = [("pʰɔː", "pʰɔː")]  # same string both sides just to test tokenization
    char_vocab = Vocab.from_strings([s for _, s in pairs], tokenizer="char")
    phon_vocab = Vocab.from_strings([s for _, s in pairs], tokenizer=TOKENIZER_PHONEME)
    char_ids = char_vocab.encode("pʰɔː")
    phon_ids = phon_vocab.encode("pʰɔː")
    # Char-level: p, ʰ, ɔ, ː = 4 tokens
    # Phoneme-level: pʰ, ɔː = 2 tokens
    assert len(phon_ids) < len(char_ids)
    assert len(phon_ids) == 2


def test_vocab_from_dict_backcompat_no_tokenizer_field():
    """Older checkpoints only stored 'itos'. Loading should default to 'char'."""
    d = {"itos": ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c"]}
    v = Vocab.from_dict(d)
    assert v.tokenizer == "char"
    assert v.encode("abc") == [4, 5, 6]


def test_vocab_to_from_dict_roundtrip_preserves_tokenizer():
    v1 = Vocab.from_strings(["pʰɔː"], tokenizer=TOKENIZER_PHONEME)
    v2 = Vocab.from_dict(v1.to_dict())
    assert v2.tokenizer == TOKENIZER_PHONEME
    # Encode goes through the phoneme tokenizer → 2 tokens, not 4.
    assert len(v2.encode("pʰɔː")) == 2
