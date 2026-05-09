"""Inference-time NeuralG2P class with greedy and beam search decoding."""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from khmer_g2p.neural.model import G2PConfig, G2PTransformer
from khmer_g2p.neural.vocab import Vocab
from khmer_g2p.normalizer import normalize


class NeuralG2P:
    """Character-level Transformer G2P for Khmer.

    Typical usage:
        g2p = NeuralG2P.from_checkpoint("checkpoints/best.pt")
        g2p.convert("ខ្មែរ")
        g2p.convert_batch(["ខ្មែរ", "ភាសា"])
        g2p.convert("ខ្មែរ", beam=4)   # beam search
    """

    def __init__(
        self,
        model: G2PTransformer,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        device: Optional[torch.device] = None,
        max_len: int = 128,
    ) -> None:
        self.model = model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device or next(model.parameters()).device
        self.max_len = max_len

    # --- Construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: Optional[str] = None,
        max_len: int = 128,
    ) -> "NeuralG2P":
        """Load a checkpoint saved by train.train()."""
        device_t = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(path, map_location=device_t, weights_only=False)
        cfg = G2PConfig.from_dict(payload["config"])
        model = G2PTransformer(cfg).to(device_t)
        model.load_state_dict(payload["model_state"])
        # load_state_dict copies into each parameter independently, which
        # silently unshares storage even when the saved checkpoint was tied.
        # Re-establish weight tying so inference matches the trained model.
        model.tie_weights_if_configured()
        src_vocab = Vocab.from_dict(payload["src_vocab"])
        tgt_vocab = Vocab.from_dict(payload["tgt_vocab"])
        return cls(model, src_vocab, tgt_vocab, device=device_t, max_len=max_len)

    # --- Public API ----------------------------------------------------------

    @torch.no_grad()
    def convert(self, word: str, beam: int = 1, length_penalty: float = 0.6) -> str:
        """Convert a single Khmer word to IPA.

        ``length_penalty`` applies the GNMT-style length normalization
        ``score / ((5 + L) / 6) ** length_penalty`` when ranking beams
        and selecting the final hypothesis. ``length_penalty=0`` recovers
        the raw sum-log-prob behavior; the default 0.6 matches GNMT.
        """
        word = normalize(word)
        if beam <= 1:
            return self._greedy(word)
        return self._beam(word, beam, length_penalty=length_penalty)

    @torch.no_grad()
    def convert_batch(
        self,
        words: List[str],
        beam: int = 1,
        length_penalty: float = 0.6,
    ) -> List[str]:
        return [self.convert(w, beam=beam, length_penalty=length_penalty) for w in words]

    # --- Greedy decode -------------------------------------------------------

    def _encode_src(self, word: str) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.src_vocab.encode(word, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        memory, src_kpm = self.model.encode(src)
        return memory, src_kpm

    def _greedy(self, word: str) -> str:
        memory, src_kpm = self._encode_src(word)
        ys = torch.tensor([[self.tgt_vocab.bos_id]],
                          dtype=torch.long, device=self.device)
        for _ in range(self.max_len):
            logits = self.model.decode(ys, memory, src_kpm)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if nxt.item() == self.tgt_vocab.eos_id:
                break
        return self.tgt_vocab.decode(ys[0].tolist()[1:])

    # --- Beam search ---------------------------------------------------------

    @staticmethod
    def _length_norm(length: int, alpha: float) -> float:
        """GNMT length-penalty divisor ``((5 + L) / 6) ** alpha``.

        Length is the number of *generated* tokens excluding BOS.
        alpha=0 disables the penalty (returns 1.0). alpha=0.6 is GNMT default.
        """
        if alpha <= 0.0:
            return 1.0
        return ((5.0 + max(1, length)) / 6.0) ** alpha

    def _beam(self, word: str, beam: int, length_penalty: float = 0.6) -> str:
        memory, src_kpm = self._encode_src(word)
        eos = self.tgt_vocab.eos_id
        bos = self.tgt_vocab.bos_id

        # Each beam entry: (raw_logprob_sum, token_ids, finished)
        # We always carry the *raw* cumulative log-prob; ranking applies the
        # length normalization on the fly so longer hypotheses aren't unfairly
        # pruned mid-search and the final pick favors length-balanced sequences.
        beams: List[Tuple[float, List[int], bool]] = [(0.0, [bos], False)]

        def _norm_score(raw: float, toks: List[int]) -> float:
            # Length excludes BOS; for an unfinished beam this is an estimate,
            # but the relative ordering is what matters during pruning.
            length = max(1, len(toks) - 1)
            return raw / self._length_norm(length, length_penalty)

        for _ in range(self.max_len):
            candidates: List[Tuple[float, List[int], bool]] = []
            # Expand every live beam
            any_live = False
            for score, toks, finished in beams:
                if finished:
                    candidates.append((score, toks, True))
                    continue
                any_live = True
                ys = torch.tensor([toks], dtype=torch.long, device=self.device)
                logits = self.model.decode(ys, memory, src_kpm)[:, -1]
                logp = torch.log_softmax(logits, dim=-1)[0]
                topk = torch.topk(logp, k=beam)
                for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                    new_toks = toks + [tok]
                    is_finished = tok == eos
                    candidates.append((score + lp, new_toks, is_finished))
            if not any_live:
                break
            # Rank by length-normalized score; keep top-K.
            candidates.sort(key=lambda x: _norm_score(x[0], x[1]), reverse=True)
            beams = candidates[:beam]
            if all(b[2] for b in beams):
                break

        best = max(beams, key=lambda b: _norm_score(b[0], b[1]))
        return self.tgt_vocab.decode(best[1][1:])  # drop BOS
