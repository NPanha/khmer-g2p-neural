"""Tests for the greedy longest-match Khmer segmenter."""

from khmer_g2p.segmenter import Segmenter


def test_segments_concatenated_words():
    # "ខ្ញុំ" + "ស្រឡាញ់" + "អ្នក"
    seg = Segmenter(vocab=["ខ្ញុំ", "ស្រឡាញ់", "អ្នក"])
    out = seg.segment("ខ្ញុំស្រឡាញ់អ្នក")
    assert out == ["ខ្ញុំ", "ស្រឡាញ់", "អ្នក"]


def test_prefers_longest_match():
    # "ខ្មែរ" should be preferred over breaking into "ខ" + rest
    seg = Segmenter(vocab=["ក", "ខ", "ខ្មែរ"])
    out = seg.segment("ខ្មែរ")
    assert out == ["ខ្មែរ"]


def test_falls_back_to_syllable_when_no_match():
    # Empty vocab → falls back to syllable-level chunking
    seg = Segmenter(vocab=[], max_word_len=10)
    out = seg.segment("ខ្ញុំ")
    # Should produce at least one chunk and reassemble to the original
    assert "".join(out) == "ខ្ញុំ"


def test_mixed_lexicon_hits_and_misses():
    seg = Segmenter(vocab=["ខ្ញុំ"])
    out = seg.segment("ខ្ញុំក")
    # 'ខ្ញុំ' is a hit; 'ក' is a syllable fallback
    assert out[0] == "ខ្ញុំ"
    assert "".join(out) == "ខ្ញុំក"


def test_vocab_contains_check():
    seg = Segmenter(vocab=["ខ្មែរ", "ភាសា"])
    assert len(seg) == 2
