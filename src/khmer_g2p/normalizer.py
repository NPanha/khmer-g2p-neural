"""Khmer text normalization and orthographic syllabification.

A Khmer orthographic syllable roughly looks like:

    [INITIAL]( COENG + SUBSCRIPT )* ( DIACRITIC | VOWEL )* [FINAL]?

where INITIAL/FINAL are base consonants, VOWEL is a dependent vowel sign, and
COENG (U+17D2) introduces a subscript consonant that stacks below the initial.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from khmer_g2p.phonemes import (
    COENG,
    DEPENDENT_VOWELS,
    INDEPENDENT_VOWELS,
    KHMER_DIGITS,
    MUUSIKATOAN,
    TRIISAP,
    BANTOC,
    ROBAT,
    TOANDAKHIAT,
    KAKABAT,
    AHSDA,
    SAMYOK_SANNYA,
    VIRIAM,
    all_consonants,
)

ZWSP = "\u200B"
ZWNJ = "\u200C"
ZWJ = "\u200D"

_DIACRITICS = {MUUSIKATOAN, TRIISAP, BANTOC, ROBAT, TOANDAKHIAT, VIRIAM, KAKABAT, AHSDA, SAMYOK_SANNYA}

# Signs that encode vowel+final in one codepoint. They may attach AFTER
# a regular dependent vowel (e.g. ុំ) and act like a final/coda, not as the
# main nuclear vowel.
_VOWEL_FINAL_SIGNS = {"\u17C6", "\u17C7", "\u17C8"}  # ំ ះ ៈ


def normalize(text: str) -> str:
    """NFC + strip zero-width chars + replace Khmer digits."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(ZWSP, "").replace(ZWNJ, "").replace(ZWJ, "")
    for khm_digit, ascii_digit in KHMER_DIGITS.items():
        text = text.replace(khm_digit, ascii_digit)
    return text


@dataclass
class Syllable:
    """A parsed Khmer orthographic syllable."""
    initial: str = ""                      # base consonant OR independent vowel
    subscripts: List[str] = field(default_factory=list)  # stacked consonants (post-COENG)
    vowel: str = ""                        # dependent vowel sign (may be empty → inherent vowel)
    vowel_final_sign: str = ""             # trailing ំ ះ ៈ (vowel+coda in one sign)
    final: str = ""                        # final consonant (bare, no COENG)
    diacritics: List[str] = field(default_factory=list)  # BANTOC, ROBAT, etc.
    raw: str = ""                          # original substring

    @property
    def is_independent_vowel(self) -> bool:
        return self.initial in INDEPENDENT_VOWELS


_CONS = all_consonants()


def syllabify(text: str) -> List[Syllable]:
    """Split a normalized Khmer string into Syllable objects.

    Non-Khmer runs (ASCII, punctuation, spaces) are emitted as Syllable objects
    with .raw set and all other fields empty — the G2P layer can pass them
    through verbatim.
    """
    text = normalize(text)
    syllables: List[Syllable] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Non-Khmer passthrough (ASCII, whitespace, punctuation, digits, etc.)
        if not ("\u1780" <= ch <= "\u17FF"):
            j = i
            while j < n and not ("\u1780" <= text[j] <= "\u17FF"):
                j += 1
            syllables.append(Syllable(raw=text[i:j]))
            i = j
            continue

        start = i
        syl = Syllable()

        # Independent vowel → standalone syllable
        if ch in INDEPENDENT_VOWELS:
            syl.initial = ch
            i += 1
            # Still allow a trailing diacritic/vowel to attach
            while i < n:
                c = text[i]
                if c in DEPENDENT_VOWELS and not syl.vowel:
                    syl.vowel = c
                    i += 1
                elif c in _DIACRITICS:
                    syl.diacritics.append(c)
                    i += 1
                else:
                    break
            syl.raw = text[start:i]
            syllables.append(syl)
            continue

        # Initial base consonant
        if ch in _CONS:
            syl.initial = ch
            i += 1
        else:
            # Orphan codepoint (e.g. stray vowel) — emit as raw passthrough
            syllables.append(Syllable(raw=ch))
            i += 1
            continue

        # Subscript consonants: (COENG + CONS)*
        while i + 1 < n and text[i] == COENG and text[i + 1] in _CONS:
            syl.subscripts.append(text[i + 1])
            i += 2

        # Dependent vowel (at most one main nuclear vowel). Vowel-final signs
        # (ំ ះ ៈ) are handled separately below so they can attach after another.
        if i < n and text[i] in DEPENDENT_VOWELS and text[i] not in _VOWEL_FINAL_SIGNS:
            syl.vowel = text[i]
            i += 1

        # Vowel-final sign (ំ ះ ៈ) — encodes vowel+coda in one codepoint
        if i < n and text[i] in _VOWEL_FINAL_SIGNS:
            syl.vowel_final_sign = text[i]
            i += 1

        # Final consonant: a trailing base consonant NOT followed by another
        # vowel or COENG is interpreted as the final.
        if (
            i < n
            and text[i] in _CONS
            and (
                i + 1 >= n
                or (text[i + 1] not in DEPENDENT_VOWELS and text[i + 1] != COENG)
            )
        ):
            # Heuristic: only consume as final if no vowel was seen yet OR a
            # vowel WAS seen. A base consonant with no vowel before it would
            # itself be a new syllable; decide by checking whether we've had
            # any vowel/subscript to "close" the initial.
            if syl.vowel or syl.subscripts:
                syl.final = text[i]
                i += 1

        # Trailing diacritics (BANTOC, TOANDAKHIAT, etc.)
        while i < n and text[i] in _DIACRITICS:
            syl.diacritics.append(text[i])
            i += 1

        # Possible vowel-encoded final (ំ, ះ already handled as vowels above —
        # nothing to do).

        syl.raw = text[start:i]
        syllables.append(syl)

    return syllables


# --- Tokenization (whitespace/punctuation boundaries) -----------------------

_TOKEN_RE = re.compile(r"([\u1780-\u17FF]+|[^\u1780-\u17FF]+)")


def tokenize(text: str) -> List[str]:
    """Split text into alternating Khmer / non-Khmer runs."""
    text = normalize(text)
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]
