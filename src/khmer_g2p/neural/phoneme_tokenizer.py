"""Khmer phoneme-level tokenizer.

Two notation systems are in use
--------------------------------
**Standard IPA** (proper Unicode): uses the length mark 'ː' (U+02D0) and the
aspiration mark 'ʰ' (U+02B0) as modifier chars, so 'pʰ' and 'aː' are each a
single token. This is what the tokenizer was *designed* for.

**Training-data notation** (the actual lexicon): uses ASCII digraphs and
doubled vowels instead, with phonemes separated by spaces:
  - Aspiration → 'ph', 'th', 'kh', 'ch'  (two chars, both split by tokenizer)
  - Long vowels → 'aa', 'ii', 'oo', 'ee', 'uu', 'ɑɑ', 'əə', 'ɔɔ', 'ɛɛ', 'ɨɨ'
  - Diphthongs  → 'ie', 'ea', 'oa', 'uə', 'iə', 'ao', 'aə', 'ɨə', 'ɛə'
  - Syllable boundary → '.'     Phoneme separator → ' ' (space)

Because the training data has no ː/ʰ modifiers, the tokenizer effectively
acts as a character splitter for that data: 'ph' → ['p', 'h'], 'aa' → ['a','a'].

Rule (applies to both notations)
---------------------------------
A phoneme token = base char + any trailing modifier chars from MODIFIERS.
All other characters are their own token.

MODIFIERS = {'ː', 'ʰ'}

Examples (standard IPA notation)
---------------------------------
    'pʰɔː'   → ['pʰ', 'ɔː']
    'ˈkeː'   → ['ˈ', 'k', 'eː']
    'kɑː cɔː'→ ['k', 'ɑː', ' ', 'c', 'ɔː']

Examples (training-data notation)
----------------------------------
    'k ɑɑ'        → ['k', ' ', 'ɑ', 'ɑ']
    'ph oa n'     → ['p', 'h', ' ', 'o', 'a', ' ', 'n']
    'k aa . r ii' → ['k', ' ', 'a', 'a', ' ', '.', ' ', 'r', ' ', 'i', 'i']
"""

from __future__ import annotations

from typing import Iterable, List, Set

# Combining modifier letters that attach to the preceding base.
#   ː  U+02D0  MODIFIER LETTER TRIANGULAR COLON (length)
#   ʰ  U+02B0  MODIFIER LETTER SMALL H          (aspiration)
MODIFIERS: Set[str] = {"\u02D0", "\u02B0"}

# Reference inventory of valid phoneme tokens.
# Used by flag_unknown() / audit_lexicon.py to detect typos or out-of-band chars.
# Covers BOTH notation systems (see module docstring).
#
# Char-level tokens (produced by the tokenizer for training-data notation):
#   Every ASCII letter and IPA base char that is a valid phoneme constituent.
# Multi-char tokens (produced when ʰ/ː modifiers are present — proper IPA only):
#   'pʰ', 'tʰ', 'kʰ', 'cʰ', 'aː', 'eː', …
# Space-separated phoneme units in training-data notation (NOT tokenizer output,
# but listed here so external callers can validate raw IPA strings):
#   aspiration digraphs: 'ph', 'th', 'kh', 'ch'
#   long vowel pairs:    'aa', 'ii', 'oo', 'ee', 'uu', 'ɑɑ', 'əə', 'ɔɔ', 'ɛɛ', 'ɨɨ'
#   diphthongs:          'ie', 'ea', 'oa', 'uə', 'iə', 'ao', 'aə', 'ɨə', 'ɛə'
KHMER_IPA_INVENTORY: Set[str] = {
    # --- Char-level consonant tokens (both notations) ---
    "p", "t", "k", "c", "b", "d", "g", "ʔ",
    "ɓ", "ɗ",                          # Khmer implosives
    "ɡ",                               # IPA script-g (U+0261), alt for 'g'
    "m", "n", "ɲ", "ŋ",
    "s", "h", "l", "r", "w", "j", "f", "v", "ʋ",
    "z",                               # loanword phoneme

    # --- Char-level vowel tokens ---
    "a", "e", "i", "o", "u",
    "ɔ", "ɑ", "ə", "ɛ", "ɨ",
    "ă", "ĕ", "ĭ", "ŏ", "ŭ",          # breve (short/reduced) vowels

    # --- Proper-IPA multi-char tokens (ʰ/ː modifiers glued by tokenizer) ---
    "pʰ", "tʰ", "kʰ", "cʰ",
    "aː", "eː", "iː", "oː", "uː",
    "ɔː", "ɑː", "əː", "ɛː", "ɨː",

    # --- Training-data aspiration digraphs (two chars, space-separated unit) ---
    "ph", "th", "kh", "ch",

    # --- Training-data long vowels (doubled, space-separated unit) ---
    "aa", "ii", "oo", "ee", "uu",
    "ɑɑ", "əə", "ɔɔ", "ɛɛ", "ɨɨ",

    # --- Training-data diphthongs / vowel clusters (space-separated unit) ---
    "ie", "ea", "oa", "ae",
    "uə", "iə", "ao", "aə", "ɨə", "ɛə",

    # --- Suprasegmentals / separators ---
    "ˈ", ".", " ",
}


def tokenize(ipa: str) -> List[str]:
    """Split an IPA string into phoneme-level tokens.

    Greedy scan: emit each base character plus any MODIFIERS that follow it.
    """
    out: List[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        ch = ipa[i]
        # Stray modifier with no base — keep as its own token so we don't
        # silently drop it (caller can inspect the result).
        if ch in MODIFIERS and not out:
            out.append(ch)
            i += 1
            continue
        # Base char + any modifiers that follow.
        j = i + 1
        while j < n and ipa[j] in MODIFIERS:
            j += 1
        out.append(ipa[i:j])
        i = j
    return out


def detokenize(tokens: Iterable[str]) -> str:
    """Join phoneme tokens back into an IPA string (lossless inverse)."""
    return "".join(tokens)


def is_known(token: str) -> bool:
    """Is this token in the reference Khmer IPA inventory?"""
    return token in KHMER_IPA_INVENTORY


def flag_unknown(tokens: Iterable[str]) -> List[str]:
    """Return the subset of tokens NOT in the reference inventory.

    Useful for auditing a lexicon — anything in here is either a rare
    phoneme or a typo.
    """
    return [t for t in tokens if not is_known(t)]
