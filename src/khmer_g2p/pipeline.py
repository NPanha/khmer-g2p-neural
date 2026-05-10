"""End-to-end Khmer text → phoneme pipeline for TTS training.

Wraps the full inference stack into one object so callers get from raw
sentence-level Khmer to a phoneme string in a single call:

    raw text
      → NFC normalize
      → tokenize into Khmer / non-Khmer runs
      → segment Khmer runs into words (lexicon-backed greedy match)
      → G2P each word (lexicon-first → neural fallback)
      → join with a configurable word-boundary token
      → preserve non-Khmer runs (digits, punctuation, ASCII) verbatim

Output format
-------------
By default the pipeline produces space-separated phoneme groups with ``|``
between words and ``||`` between sentences-as-passed-in (only relevant for
``phonemize_batch``):

    "ខ្មែរ ភាសា"   →   "kʰ m ae r | pʰ ie s aa"

This matches the lexicon's own notation (e.g. ``kʰ m ae r`` in
``data/lexicon.tsv``), so the same downstream tokenizer can split on
spaces during TTS data loading.

Typical usage
-------------

    from khmer_g2p.pipeline import KhmerG2PPipeline

    pipe = KhmerG2PPipeline.from_paths(
        ckpt="checkpoints_v04/best.pt",
        lexicon="data/lexicon.tsv",
    )
    pipe.phonemize("ខ្ញុំស្រឡាញ់ភាសាខ្មែរ")
    # → "kʰ ɲ om | s rɑ l a ɲ | pʰ ie s aa | kʰ m ae r"

    # In a TTS data loader:
    phoneme_str = pipe.phonemize(metadata_text)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from khmer_g2p.lexicon import Lexicon, load_tsv
from khmer_g2p.normalizer import normalize, tokenize
from khmer_g2p.segmenter import Segmenter

if TYPE_CHECKING:  # avoid pulling torch at import time
    from khmer_g2p.hybrid import HybridG2P
    from khmer_g2p.neural.infer import NeuralG2P


_KHMER_RANGE = ("ក", "៿")


def _is_khmer_run(s: str) -> bool:
    return bool(s) and _KHMER_RANGE[0] <= s[0] <= _KHMER_RANGE[1]


@dataclass
class PipelineConfig:
    """Knobs for joining and decoding behavior."""

    word_boundary: str = "|"      # token inserted between Khmer words
    phoneme_sep: str = " "         # separator between phoneme groups
    beam: int = 4                  # beam width for neural fallback
    length_penalty: float = 0.6    # GNMT-style length norm for beam
    keep_non_khmer: bool = True    # pass through ASCII / punctuation
    cache_words: bool = True       # memoize per-word G2P results


class KhmerG2P:
    """Sentence-level Khmer text → phoneme string pipeline.

    The pipeline owns three stages:

    - ``segmenter``: greedy longest-match Khmer word segmenter built from
      the lexicon vocabulary.
    - ``g2p``: :class:`HybridG2P` — lexicon-first lookup with neural fallback.
    - per-word cache: avoids re-running the model on repeated tokens within
      a corpus (e.g. function words like ``និង``, ``នៅ``, …).
    """

    def __init__(
        self,
        g2p: "HybridG2P",
        segmenter: Segmenter,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.g2p = g2p
        self.segmenter = segmenter
        self.config = config or PipelineConfig()
        self._cache: Dict[str, str] = {}

    # --- Construction --------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        ckpt: str | Path,
        lexicon: str | Path,
        device: Optional[str] = None,
        max_len: int = 128,
        config: Optional[PipelineConfig] = None,
    ) -> "KhmerG2PPipeline":
        """Build a pipeline from a checkpoint and TSV lexicon."""
        from khmer_g2p.hybrid import HybridG2P
        from khmer_g2p.neural.infer import NeuralG2P

        neural = NeuralG2P.from_checkpoint(ckpt, device=device, max_len=max_len)
        lex = Lexicon(load_tsv(lexicon))
        hybrid = HybridG2P(neural, lex)
        seg = Segmenter(lex.words())
        return cls(hybrid, seg, config=config)

    @classmethod
    def from_components(
        cls,
        neural: "NeuralG2P",
        lexicon: Lexicon,
        config: Optional[PipelineConfig] = None,
    ) -> "KhmerG2P":
        """Build from an already-loaded neural model and lexicon."""
        from khmer_g2p.hybrid import HybridG2P

        hybrid = HybridG2P(neural, lexicon)
        seg = Segmenter(lexicon.words())
        return cls(hybrid, seg, config=config)

    # --- Public API ----------------------------------------------------------

    def phonemize(self, text: str) -> str:
        """Convert one Khmer text string to a phoneme sequence.

        Khmer runs are segmented into words and G2P'd; non-Khmer runs are
        passed through verbatim (subject to ``config.keep_non_khmer``).
        Whitespace between user-supplied tokens is treated as a word break
        and emitted as a single ``word_boundary`` marker.
        """
        cfg = self.config
        text = normalize(text)
        if not text:
            return ""

        # Each entry is either a phoneme group (already internally
        # space-separated) or a non-Khmer chunk to pass through verbatim.
        # ``None`` marks a word boundary; collapsed at the end.
        parts: List[Optional[str]] = []

        def push(value: Optional[str]) -> None:
            if value is None:
                if parts and parts[-1] is not None:
                    parts.append(None)
                return
            parts.append(value)

        for run in tokenize(text):
            if _is_khmer_run(run):
                for word in self.segmenter.segment(run):
                    if not word:
                        continue
                    push(self._phonemize_word(word))
                    push(None)
            else:
                stripped = run.strip()
                if stripped and cfg.keep_non_khmer:
                    push(stripped)
                    push(None)
                elif any(ch.isspace() for ch in run):
                    push(None)

        # Strip trailing None, then materialize to strings.
        while parts and parts[-1] is None:
            parts.pop()

        rendered = [cfg.word_boundary if p is None else p for p in parts]
        return " ".join(rendered)

    def phonemize_batch(self, texts: Iterable[str]) -> List[str]:
        """Phonemize a list of strings. Order-preserving."""
        return [self.phonemize(t) for t in texts]

    def khm_phonemize(self, text: str) -> str:
        """Phonemize and return word-joined phonemes ready for TTS training.

        Phonemes within a word are concatenated (no space), syllable dots are
        dropped, and words are separated by a single space.
        """
        s = self.phonemize(text)
        wb = self.config.word_boundary or "|"
        words = [w.strip() for w in s.split(f" {wb} ")]
        return " ".join(
            "".join(w.replace(".", " ").split()) for w in words if w
        )

    # --- Internals -----------------------------------------------------------

    def _phonemize_word(self, word: str) -> str:
        """G2P one already-segmented Khmer word, with caching."""
        cfg = self.config
        if cfg.cache_words and word in self._cache:
            return self._cache[word]
        ipa = self.g2p.convert(
            word,
            beam=cfg.beam,
            length_penalty=cfg.length_penalty,
        )
        # Normalize internal whitespace to the configured phoneme separator.
        # The lexicon already uses single spaces, the model decoder reproduces
        # them; this guard keeps multi-space artifacts out.
        ipa = cfg.phoneme_sep.join(ipa.split())
        if cfg.cache_words:
            self._cache[word] = ipa
        return ipa

    # --- Diagnostics ---------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Lexicon hit rate (lifetime of underlying HybridG2P)."""
        return self.g2p.hit_rate

    def reset_stats(self) -> None:
        """Clear lexicon hit/miss counters and per-word cache."""
        self.g2p.reset_stats()
        self._cache.clear()

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"KhmerG2PPipeline("
            f"lexicon_size={len(self.g2p)}, "
            f"segmenter_vocab={len(self.segmenter)}, "
            f"beam={cfg.beam}, "
            f"word_boundary={cfg.word_boundary!r}, "
            f"hit_rate={self.hit_rate:.3f})"
        )
