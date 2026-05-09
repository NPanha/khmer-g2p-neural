"""Lexicon loader and in-memory lookup.

A Khmer pronunciation lexicon is stored as TSV:

    KHMER_WORD<TAB>IPA

This module provides:

- ``load_tsv``  : parse a TSV file into a list of ``(word, ipa)`` pairs.
- ``Lexicon``   : dict-backed lookup with multi-variant support.

No FST / pynini dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from khmer_g2p.normalizer import normalize


LexiconEntry = Tuple[str, str]

_HEADER_TOKENS = {"word", "grapheme", "graphemes", "orthography", "text"}
_IPA_HEADER_TOKENS = {"ipa", "pronunciation", "phonemes", "phones"}


def _looks_like_header(word: str, ipa: str) -> bool:
    return (
        word.strip().lower() in _HEADER_TOKENS
        and ipa.strip().lower() in _IPA_HEADER_TOKENS
    )


def load_tsv(path: str | Path) -> List[LexiconEntry]:
    """Load a two-column TSV lexicon.

    - Blank lines and '#' comments are skipped.
    - A leading header row (``word<TAB>ipa`` or similar) is auto-detected and skipped.
    - Multi-variant entries (same word, multiple rows) are all preserved; the
      first variant wins for dict lookup while every variant is kept in
      ``Lexicon.variants``.
    - Leading/trailing whitespace is stripped from the IPA; leading dots (used
      as syllable separators in some sources) are preserved as-is.
    - ``utf-8-sig`` transparently strips a leading BOM (U+FEFF) if present.
    """
    entries: List[LexiconEntry] = []
    p = Path(path)
    with p.open(encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = stripped.split(None, 1)
            if len(parts) < 2:
                raise ValueError(f"{p}:{line_no}: expected '<word>\\t<ipa>'")
            word = normalize(parts[0].strip())
            ipa = parts[1].strip()
            if _looks_like_header(word, ipa):
                continue
            if word and ipa:
                entries.append((word, ipa))
    return entries


class Lexicon:
    """Dict-backed lookup with multi-variant support.

    Multi-variant entries are preserved: ``get()`` returns the first-seen
    pronunciation for a word, while ``variants()`` returns all of them.
    """

    def __init__(self, entries: Optional[Iterable[LexiconEntry]] = None) -> None:
        entries = list(entries) if entries else []
        self._variants: Dict[str, List[str]] = {}
        for w, p in entries:
            self._variants.setdefault(w, []).append(p)
        self._raw_count = len(entries)

    @classmethod
    def from_tsv(cls, path: str | Path) -> "Lexicon":
        return cls(load_tsv(path))

    def __contains__(self, word: str) -> bool:
        return normalize(word) in self._variants

    def __len__(self) -> int:
        """Number of distinct words (not entries; use ``entry_count`` for rows)."""
        return len(self._variants)

    @property
    def entry_count(self) -> int:
        """Total number of (word, ipa) rows, counting variants separately."""
        return self._raw_count

    def words(self) -> List[str]:
        """Return the list of distinct Khmer words in the lexicon."""
        return list(self._variants.keys())

    def get(self, word: str) -> Optional[str]:
        """Return the first (canonical) pronunciation for ``word``, or None."""
        vs = self._variants.get(normalize(word))
        return vs[0] if vs else None

    def variants(self, word: str) -> List[str]:
        """Return all recorded pronunciations for ``word`` (possibly empty)."""
        return list(self._variants.get(normalize(word), []))
