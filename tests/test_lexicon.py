"""Tests for the dict-backed lexicon."""

from khmer_g2p.lexicon import Lexicon


def test_lexicon_basic_lookup():
    entries = [("ខ្ញុំ", "kʰɲom"), ("អ្នក", "neək")]
    lex = Lexicon(entries)

    assert lex.get("ខ្ញុំ") == "kʰɲom"
    assert lex.get("អ្នក") == "neək"
    assert lex.get("???") is None
    assert "ខ្ញុំ" in lex
    assert "???" not in lex


def test_empty_lexicon_is_safe():
    lex = Lexicon([])
    assert lex.get("anything") is None
    assert len(lex) == 0


def test_multi_variant_entries():
    entries = [
        ("ឧបនីយកម្ម", ".ʔuː.paʔ.niː.jĕəʔ.kam"),
        ("ឧបនីយកម្ម", ".ʔuː.pə.niː.jə.kam"),
    ]
    lex = Lexicon(entries)
    # First seen wins for .get()
    assert lex.get("ឧបនីយកម្ម") == ".ʔuː.paʔ.niː.jĕəʔ.kam"
    # Both variants are preserved
    assert len(lex.variants("ឧបនីយកម្ម")) == 2
    assert len(lex) == 1
    assert lex.entry_count == 2


def test_header_row_is_skipped(tmp_path):
    from khmer_g2p.lexicon import load_tsv
    p = tmp_path / "lex.tsv"
    p.write_text("word\tipa\nក\tkɑː\n", encoding="utf-8")
    entries = load_tsv(p)
    assert entries == [("ក", "kɑː")]
