"""Vocabulary for source (Khmer) and target (IPA).

Supports two tokenization modes:

- ``"char"``  — one Unicode codepoint per token (original behavior).
- ``"phoneme"`` — IPA phonemes (via ``phoneme_tokenizer.tokenize``),
  where 'pʰ' and 'aː' are single tokens. Only valid on the target side.

The tokenization mode is stored on the Vocab itself so a checkpoint knows
how to re-encode/decode at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from khmer_g2p.neural.phoneme_tokenizer import tokenize as phoneme_tokenize

# Special tokens
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS: Tuple[str, ...] = (PAD, BOS, EOS, UNK)

# Valid tokenization modes
TOKENIZER_CHAR = "char"
TOKENIZER_PHONEME = "phoneme"


def _tokenize(text: str, mode: str) -> List[str]:
    """Split `text` into tokens according to `mode`."""
    if mode == TOKENIZER_CHAR:
        return list(text)
    if mode == TOKENIZER_PHONEME:
        return phoneme_tokenize(text)
    raise ValueError(f"Unknown tokenizer mode: {mode!r}")


@dataclass
class Vocab:
    """Token vocabulary with reserved special tokens.

    Attributes
    ----------
    itos : list[str]       index → symbol
    stoi : dict[str, int]  symbol → index
    tokenizer : str        "char" or "phoneme" — how to split strings
    """
    itos: List[str] = field(default_factory=list)
    stoi: Dict[str, int] = field(default_factory=dict)
    tokenizer: str = TOKENIZER_CHAR

    # --- Build -----------------------------------------------------------------

    @classmethod
    def from_tokens(cls, tokens: Iterable[str], tokenizer: str = TOKENIZER_CHAR) -> "Vocab":
        """Build a vocab from an iterable of already-split tokens.

        Deterministic ordering: specials first, then sorted unique tokens.
        """
        seen: set = set()
        for t in tokens:
            seen.add(t)
        ordered_sorted = sorted(seen)
        itos = list(SPECIALS) + [t for t in ordered_sorted if t not in SPECIALS]
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(itos=itos, stoi=stoi, tokenizer=tokenizer)

    @classmethod
    def from_chars(cls, chars: Iterable[str]) -> "Vocab":
        """Legacy: build a char-level vocab. Equivalent to ``from_tokens(chars, 'char')``."""
        return cls.from_tokens(chars, tokenizer=TOKENIZER_CHAR)

    @classmethod
    def from_strings(cls, strings: Iterable[str], tokenizer: str = TOKENIZER_CHAR) -> "Vocab":
        """Build a vocab from full strings by tokenizing each according to ``tokenizer``."""
        tokens: List[str] = []
        for s in strings:
            tokens.extend(_tokenize(s, tokenizer))
        return cls.from_tokens(tokens, tokenizer=tokenizer)

    # --- Properties ------------------------------------------------------------

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    # --- Encode / decode -------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a string as a list of token IDs (tokenizer-aware)."""
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in _tokenize(text, self.tokenizer):
            ids.append(self.stoi.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> str:
        """Decode a list of token IDs back to a string.

        Tokens are concatenated directly — this is the inverse of ``encode``
        for both 'char' and 'phoneme' modes (phoneme tokens like 'pʰ' join
        losslessly since they were never split internally).
        """
        out: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            sym = self.itos[i]
            if strip_specials and sym in SPECIALS:
                if sym == EOS:
                    break
                continue
            out.append(sym)
        return "".join(out)

    # --- Serialize -------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {"itos": self.itos, "tokenizer": self.tokenizer}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Vocab":
        itos = list(d["itos"])  # type: ignore[arg-type]
        stoi = {s: i for i, s in enumerate(itos)}
        # Back-compat: older checkpoints stored only {"itos": [...]}
        tokenizer = d.get("tokenizer", TOKENIZER_CHAR)  # type: ignore[assignment]
        return cls(itos=itos, stoi=stoi, tokenizer=str(tokenizer))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)


def build_vocabs(
    pairs: Iterable[Tuple[str, str]],
    src_tokenizer: str = TOKENIZER_CHAR,
    tgt_tokenizer: str = TOKENIZER_CHAR,
) -> Tuple[Vocab, Vocab]:
    """Build (source_vocab, target_vocab) from (word, ipa) pairs.

    Defaults to char-level for both sides (back-compat). Pass
    ``tgt_tokenizer="phoneme"`` to tokenize the IPA target into phonemes.
    Source (Khmer) should stay char-level — Khmer orthography has no
    meaningful phoneme-analogue.
    """
    src_tokens: List[str] = []
    tgt_tokens: List[str] = []
    for w, p in pairs:
        src_tokens.extend(_tokenize(w, src_tokenizer))
        tgt_tokens.extend(_tokenize(p, tgt_tokenizer))
    return (
        Vocab.from_tokens(src_tokens, tokenizer=src_tokenizer),
        Vocab.from_tokens(tgt_tokens, tokenizer=tgt_tokenizer),
    )
