"""Multi-seed ensemble decoding.

Train N models with different ``--seed`` values, then average their *log*
probabilities at every decoding step. With N≈3–5 this is the single largest
typical jump after the v0.4 recipe — the seeds disagree most on the long-tail
phonemes, so averaging the disagreement away is essentially free PER.

Usage::

    from khmer_g2p.neural.ensemble import EnsembleG2P

    g2p = EnsembleG2P.from_checkpoints(
        ["checkpoints_s42/best.pt",
         "checkpoints_s7/best.pt",
         "checkpoints_s2025/best.pt"],
    )
    g2p.convert("ខ្មែរ", beam=4, length_penalty=0.6)

The class API mirrors :class:`khmer_g2p.neural.infer.NeuralG2P` and
:class:`khmer_g2p.hybrid.HybridG2P` (same ``convert``/``convert_batch``
signatures), so it slots into ``live_test.py`` and any other inference
plumbing without changes.

Constraints
-----------
- All ensemble members must share the **same** source and target vocabularies.
  The constructor raises if they don't.
- Beam search applies the length penalty to the averaged log-probs (the
  standard formulation), not to each member individually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from khmer_g2p.neural.infer import NeuralG2P
from khmer_g2p.normalizer import normalize


class EnsembleG2P:
    """Average per-step log-probs across N independently trained G2P models."""

    def __init__(
        self,
        members: Sequence[NeuralG2P],
        max_len: int = 128,
    ) -> None:
        if len(members) < 2:
            raise ValueError("ensemble needs at least 2 members")
        self.members: List[NeuralG2P] = list(members)
        # All members must agree on vocabularies — otherwise averaging logits
        # makes no sense (different positions encode different tokens).
        ref = self.members[0]
        for i, m in enumerate(self.members[1:], start=1):
            if m.src_vocab.itos != ref.src_vocab.itos:
                raise ValueError(
                    f"member {i}: src_vocab differs from member 0 — "
                    "all ensemble members must share the same src_vocab."
                )
            if m.tgt_vocab.itos != ref.tgt_vocab.itos:
                raise ValueError(
                    f"member {i}: tgt_vocab differs from member 0 — "
                    "all ensemble members must share the same tgt_vocab."
                )
        self.src_vocab = ref.src_vocab
        self.tgt_vocab = ref.tgt_vocab
        self.device = ref.device
        self.max_len = max_len

    # --- Construction -------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        paths: Iterable[str | Path],
        device: Optional[str] = None,
        max_len: int = 128,
    ) -> "EnsembleG2P":
        """Load each path through ``NeuralG2P.from_checkpoint`` then ensemble."""
        members = [
            NeuralG2P.from_checkpoint(p, device=device, max_len=max_len)
            for p in paths
        ]
        return cls(members, max_len=max_len)

    # --- Internal helpers ---------------------------------------------------

    def _encode_all(self, word: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode the source once per member, returning ``(memory, kpm)`` pairs."""
        word = normalize(word)
        out = []
        for m in self.members:
            src_ids = m.src_vocab.encode(word, add_eos=True)
            src = torch.tensor([src_ids], dtype=torch.long, device=m.device)
            mem, kpm = m.model.encode(src)
            out.append((mem, kpm))
        return out

    @torch.no_grad()
    def _step_log_probs(
        self,
        ys: torch.Tensor,
        memos: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Average decoder log-probs from every ensemble member at the last step.

        Returns a tensor of shape (V,) on ``self.device``.
        """
        accum: Optional[torch.Tensor] = None
        for member, (mem, kpm) in zip(self.members, memos):
            ys_m = ys.to(member.device)
            logits = member.model.decode(ys_m, mem, kpm)[:, -1]              # (1, V)
            lp = torch.log_softmax(logits, dim=-1)[0]                         # (V,)
            if accum is None:
                accum = lp.to(self.device)
            else:
                accum = accum + lp.to(self.device)
        accum = accum / len(self.members)
        return accum

    # --- Public API ---------------------------------------------------------

    @torch.no_grad()
    def convert(self, word: str, beam: int = 1, length_penalty: float = 0.6) -> str:
        memos = self._encode_all(word)
        if beam <= 1:
            return self._greedy(memos)
        return self._beam(memos, beam, length_penalty=length_penalty)

    @torch.no_grad()
    def convert_batch(
        self,
        words: List[str],
        beam: int = 1,
        length_penalty: float = 0.6,
    ) -> List[str]:
        return [self.convert(w, beam=beam, length_penalty=length_penalty) for w in words]

    # --- Greedy / beam ------------------------------------------------------

    def _greedy(self, memos) -> str:
        bos = self.tgt_vocab.bos_id
        eos = self.tgt_vocab.eos_id
        ys = torch.tensor([[bos]], dtype=torch.long, device=self.device)
        for _ in range(self.max_len):
            lp = self._step_log_probs(ys, memos)
            nxt = int(lp.argmax().item())
            ys = torch.cat([ys, torch.tensor([[nxt]], dtype=torch.long,
                                             device=self.device)], dim=1)
            if nxt == eos:
                break
        return self.tgt_vocab.decode(ys[0].tolist()[1:])

    def _beam(self, memos, beam: int, length_penalty: float) -> str:
        bos = self.tgt_vocab.bos_id
        eos = self.tgt_vocab.eos_id
        beams: List[Tuple[float, List[int], bool]] = [(0.0, [bos], False)]

        def _norm(raw: float, toks: List[int]) -> float:
            length = max(1, len(toks) - 1)
            div = NeuralG2P._length_norm(length, length_penalty)
            return raw / div

        for _ in range(self.max_len):
            cands: List[Tuple[float, List[int], bool]] = []
            any_live = False
            for score, toks, finished in beams:
                if finished:
                    cands.append((score, toks, True))
                    continue
                any_live = True
                ys = torch.tensor([toks], dtype=torch.long, device=self.device)
                lp = self._step_log_probs(ys, memos)
                topk = torch.topk(lp, k=beam)
                for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    cands.append((score + v, toks + [idx], idx == eos))
            if not any_live:
                break
            cands.sort(key=lambda x: _norm(x[0], x[1]), reverse=True)
            beams = cands[:beam]
            if all(b[2] for b in beams):
                break

        best = max(beams, key=lambda b: _norm(b[0], b[1]))
        return self.tgt_vocab.decode(best[1][1:])

    def __len__(self) -> int:
        return len(self.members)
