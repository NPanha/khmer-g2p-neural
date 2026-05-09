"""Per-position auxiliary labels derived from the source character sequence.

Two heads are supported, each computed from the same ``src`` tensor used by
the encoder:

* **Series head.** For every position whose token is a Khmer base consonant,
  emit class 0 (Series-1, "ɑ-series") or class 1 (Series-2, "oː-series").
  All other positions (vowels, COENG, diacritics, digits, padding, BOS/EOS)
  are marked with the ignore index ``-100`` so cross-entropy skips them.

* **Syllable boundary head.** For every Khmer character position, emit 1 if
  it starts a new orthographic syllable (per ``normalizer.syllabify``) and 0
  otherwise. Padding / BOS / EOS / non-Khmer get ``-100``.

These auxiliary objectives push the encoder toward representations that
explicitly encode the two structural properties Khmer phonology hinges on:
which series a consonant belongs to (governs vowel realization) and where
syllable boundaries fall (governs prosody and final-consonant rules).

The helpers operate on a *batched* ``src`` tensor and the matching source
``Vocab``. They use only character lookups — no model state, no extra dataset
fields needed.
"""

from __future__ import annotations

from typing import Tuple

import torch

from khmer_g2p.neural.vocab import SPECIALS as _SPECIALS_TUPLE
from khmer_g2p.normalizer import syllabify
from khmer_g2p.phonemes import CONSONANT_SERIES

_SPECIALS = set(_SPECIALS_TUPLE)
IGNORE_INDEX: int = -100

# Output class layout (kept here so the model and the trainer agree).
SERIES_CLASSES: int = 2     # 0 = series 1 (ɑ),  1 = series 2 (oː)
SYLLABLE_CLASSES: int = 2   # 0 = continuation, 1 = boundary (start of syllable)


def _row_to_chars(row_ids, src_vocab) -> Tuple[list, list]:
    """Return ``(chars, positions)`` for the non-special tokens of one row.

    ``chars[i]`` is the printable character at original position ``positions[i]``
    in the batched ``src`` tensor. Padding, BOS, EOS and any token whose symbol
    isn't a single character (i.e. specials) are skipped.
    """
    chars: list = []
    positions: list = []
    itos = src_vocab.itos
    for pos, tok_id in enumerate(row_ids):
        if tok_id == src_vocab.pad_id:
            continue
        # bos_id and eos_id might or might not be present depending on encode()
        # call; defensively skip both.
        if tok_id == src_vocab.eos_id or tok_id == src_vocab.bos_id:
            continue
        sym = itos[tok_id]
        if sym in _SPECIALS:
            continue
        if len(sym) != 1:
            # Phoneme-tokenizer sources would land here. We're a char-source
            # model in practice, but the helper degrades gracefully.
            continue
        chars.append(sym)
        positions.append(pos)
    return chars, positions


def extract_series_labels(src: torch.Tensor, src_vocab) -> torch.Tensor:
    """Per-position consonant-series labels (or IGNORE_INDEX).

    Shape ``(B, T)``; dtype ``long``; same device as ``src``.
    Class meanings: 0 = Series 1, 1 = Series 2, -100 = not a base consonant.
    """
    B, T = src.shape
    out = torch.full((B, T), IGNORE_INDEX, dtype=torch.long, device=src.device)
    src_cpu = src.detach().cpu().tolist()
    for i, row in enumerate(src_cpu):
        chars, positions = _row_to_chars(row, src_vocab)
        for pos, ch in zip(positions, chars):
            s = CONSONANT_SERIES.get(ch)
            if s == 1:
                out[i, pos] = 0
            elif s == 2:
                out[i, pos] = 1
            # else: leave as IGNORE_INDEX (vowel, diacritic, COENG, etc.)
    return out


def extract_syllable_labels(src: torch.Tensor, src_vocab) -> torch.Tensor:
    """Per-position syllable-boundary labels (or IGNORE_INDEX).

    Class 1 marks the first character of a new orthographic syllable; class 0
    marks continuation characters within a syllable. Non-Khmer / BOS / EOS /
    PAD positions get -100.
    """
    B, T = src.shape
    out = torch.full((B, T), IGNORE_INDEX, dtype=torch.long, device=src.device)
    src_cpu = src.detach().cpu().tolist()
    for i, row in enumerate(src_cpu):
        chars, positions = _row_to_chars(row, src_vocab)
        if not chars:
            continue
        text = "".join(chars)
        sylls = syllabify(text)
        char_idx = 0
        for syl in sylls:
            length = len(syl.raw)
            if length == 0:
                continue
            for j in range(length):
                k = char_idx + j
                if k >= len(positions):
                    break
                pos = positions[k]
                out[i, pos] = 1 if j == 0 else 0
            char_idx += length
            if char_idx >= len(chars):
                break
    return out


def extract_aux_labels(
    src: torch.Tensor, src_vocab
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: return ``(series_labels, syllable_labels)`` in one pass."""
    return extract_series_labels(src, src_vocab), extract_syllable_labels(src, src_vocab)
