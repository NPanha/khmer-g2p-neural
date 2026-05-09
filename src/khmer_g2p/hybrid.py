"""Hybrid lexicon-first, neural-fallback G2P.

For words that exist in the training lexicon, the canonical IPA is returned
verbatim — no model call, no risk of regression. For words that don't exist
in the lexicon, the request is forwarded to the neural model.

This eliminates the most embarrassing class of errors in deployment: words
the model was *trained on* coming back wrong because the network's
generalization happened to win out over memorization.

Typical usage::

    from khmer_g2p.hybrid import HybridG2P

    g2p = HybridG2P.from_paths(
        ckpt="checkpoints_v04/best.pt",
        lexicon="data/lexicon.tsv",
    )
    g2p.convert("ខ្មែរ")          # lexicon hit (or model fallback)
    g2p.convert("ខ្មែរ", beam=4)  # beam used only on fallback
    g2p.last_source                  # "lexicon" or "model"

The same ``HybridG2P`` instance can be used with the existing live-test and
prediction CLIs — its ``convert`` / ``convert_batch`` signature matches
:class:`khmer_g2p.neural.infer.NeuralG2P` exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from khmer_g2p.lexicon import Lexicon, load_tsv
from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.normalizer import normalize


class HybridG2P:
    """Lexicon-first G2P with a neural fallback.

    Parameters
    ----------
    neural : NeuralG2P
        The trained neural model.
    lexicon : Lexicon | dict[str, str]
        Either a :class:`khmer_g2p.lexicon.Lexicon` or a plain dict mapping
        normalized Khmer words to IPA strings. Variants beyond the first are
        ignored (use :class:`Lexicon` directly if you want all of them).
    """

    def __init__(
        self,
        neural: NeuralG2P,
        lexicon: "Lexicon | Dict[str, str]",
    ) -> None:
        self.neural = neural
        if isinstance(lexicon, Lexicon):
            self._lookup: Dict[str, str] = {
                w: lexicon.get(w) or "" for w in lexicon.words()
            }
        else:
            # Already a dict — normalize the keys to be safe.
            self._lookup = {normalize(k): v for k, v in lexicon.items()}
        # For tests / diagnostics: which path produced the last result.
        self.last_source: str = "none"
        # Counters across the lifetime of this object.
        self.hits: int = 0
        self.misses: int = 0

    # --- Construction --------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        ckpt: str | Path,
        lexicon: str | Path,
        device: Optional[str] = None,
        max_len: int = 128,
    ) -> "HybridG2P":
        """Load a checkpoint and a TSV lexicon and build a HybridG2P.

        ``lexicon`` is the same file you used to train the model. It's
        loaded with the standard ``load_tsv`` so multi-variant entries are
        handled identically — the first listed pronunciation wins on a hit.
        """
        neural = NeuralG2P.from_checkpoint(ckpt, device=device, max_len=max_len)
        lex = Lexicon(load_tsv(lexicon))
        return cls(neural, lex)

    @classmethod
    def from_pairs(
        cls,
        neural: NeuralG2P,
        pairs: Iterable[Tuple[str, str]],
    ) -> "HybridG2P":
        """Build from an iterable of ``(word, ipa)`` pairs (first variant wins)."""
        lex = Lexicon(pairs)
        return cls(neural, lex)

    # --- Public API ----------------------------------------------------------

    def convert(self, word: str, beam: int = 1, length_penalty: float = 0.6) -> str:
        """Convert a single Khmer word.

        Lookup order:
            1. Exact match against the (NFC-normalized) lexicon.
            2. Fallback to the neural model with greedy or beam decoding.

        ``beam`` and ``length_penalty`` are only used on a miss; lexicon
        hits return the stored IPA verbatim.
        """
        key = normalize(word)
        ipa = self._lookup.get(key)
        if ipa:
            self.last_source = "lexicon"
            self.hits += 1
            return ipa
        self.last_source = "model"
        self.misses += 1
        return self.neural.convert(key, beam=beam, length_penalty=length_penalty)

    def convert_batch(
        self,
        words: List[str],
        beam: int = 1,
        length_penalty: float = 0.6,
    ) -> List[str]:
        return [self.convert(w, beam=beam, length_penalty=length_penalty) for w in words]

    # --- Diagnostics ---------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Lexicon hit rate over the lifetime of this instance."""
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def __len__(self) -> int:
        """Number of unique words in the lexicon."""
        return len(self._lookup)

    def __contains__(self, word: str) -> bool:
        return normalize(word) in self._lookup

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.last_source = "none"
