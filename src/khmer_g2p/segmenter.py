"""Khmer word segmentation via greedy longest-match against a lexicon.

Khmer writing has no spaces between words, so any sentence-level input has to
be split into words before G2P can process each one. This module implements
*forward maximum-matching* (a.k.a. longest-prefix matching): at each position,
pick the longest prefix that appears in the lexicon as a word. If nothing
matches, peel off one orthographic syllable (via the syllabifier) and move
on. The result is a list of word-like units that the G2P pipeline can
process independently.

This is the simplest competent Khmer segmenter. It works well when the
lexicon has good coverage of common words (which yours, at ~10k entries,
does for most everyday text). For open-domain text or domain-specific jargon
consider pairing with a dedicated Khmer tokenizer (e.g. khmer-nltk).
"""

from __future__ import annotations

from typing import Iterable, List, Set

from khmer_g2p.normalizer import normalize, syllabify


class Segmenter:
    """Greedy longest-match Khmer word segmenter."""

    def __init__(
        self,
        vocab: Iterable[str],
        max_word_len: int | None = None,
    ) -> None:
        self.vocab: Set[str] = {normalize(w) for w in vocab if w}
        # Cap the search window at the longest known word to keep matching fast.
        if max_word_len is None:
            max_word_len = max((len(w) for w in self.vocab), default=20)
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.vocab)

    def segment(self, text: str) -> List[str]:
        """Split a Khmer string into a list of word-like units.

        Non-Khmer characters (ASCII, whitespace, punctuation) are *not*
        expected here — the caller should pass a single Khmer run.
        """
        text = normalize(text)
        out: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            match_len = 0
            # Try progressively shorter prefixes starting at position i.
            upper = min(self.max_word_len, n - i)
            for L in range(upper, 0, -1):
                if text[i : i + L] in self.vocab:
                    match_len = L
                    break

            if match_len > 0:
                out.append(text[i : i + match_len])
                i += match_len
                continue

            # No lexicon match. Fall back to one orthographic syllable.
            rest_syls = syllabify(text[i:])
            if rest_syls and rest_syls[0].raw:
                seg = rest_syls[0].raw
                out.append(seg)
                i += len(seg)
            else:
                # Defensive fallback (shouldn't trigger in practice).
                out.append(text[i])
                i += 1

        return out
