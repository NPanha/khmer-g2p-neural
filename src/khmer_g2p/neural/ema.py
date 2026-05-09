"""Exponential moving average of model weights.

A shadow copy of the model parameters that gets nudged toward the live model
after every optimizer step::

    shadow = decay * shadow + (1 - decay) * live

Evaluating on the EMA weights (and saving them as ``best.pt``) is one of the
cheapest "free" wins in seq2seq training — it smooths over the optimizer's
late-stage zigzagging and typically buys 0.5–1.5 PER on G2P.

We follow the timm convention of gradually ramping the decay during early
training so the shadow isn't dominated by the random init::

    effective_decay = min(decay, (1 + step) / (10 + step))

so it starts near 0.09 at step 1 and asymptotes to ``decay``. By the time the
LR-warmup window closes, the EMA is already close to its target decay.

Usage::

    ema = ModelEma(model, decay=0.999)
    for step, batch in enumerate(loader, 1):
        ...
        optim.step()
        ema.update(model, step=step)

    # Evaluate on the EMA weights, then bring the live ones back.
    with ema.apply_to(model):
        evaluate(model, ...)

    # Save EMA as best.pt.
    torch.save({"model_state": ema.state_dict(), ...}, best_path)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class ModelEma:
    """Maintain a polyak-averaged shadow copy of a model's state_dict.

    Floating-point tensors are stored in float32 regardless of the live
    model's dtype (so AMP / bf16 training still gets a clean EMA). Buffers
    and integer tensors are simply tracked by reference-copy, since they
    don't have a meaningful average.

    Parameters
    ----------
    model : nn.Module
    decay : float
        Target EMA decay. 0.999 is a sensible default for seq2seq training
        with batch sizes in the dozens; bump to 0.9999 for very large batches.
    use_warmup : bool
        Apply the timm-style ``min(decay, (1 + step) / (10 + step))`` ramp.
        Recommended on (default).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        use_warmup: bool = True,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = float(decay)
        self.use_warmup = bool(use_warmup)
        self._shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self._shadow[k] = v.detach().clone().float()
            else:
                self._shadow[k] = v.detach().clone()

    # --- core update --------------------------------------------------------

    def _effective_decay(self, step: int) -> float:
        if not self.use_warmup or step <= 0:
            return self.decay
        ramp = (1.0 + step) / (10.0 + step)
        return min(self.decay, ramp)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int = 0) -> None:
        """Pull the live model's parameters one EMA step closer to the shadow."""
        d = self._effective_decay(step)
        for k, v in model.state_dict().items():
            if k not in self._shadow:
                # New parameter (e.g. a head attached mid-training) — adopt it.
                self._shadow[k] = (
                    v.detach().clone().float()
                    if v.is_floating_point()
                    else v.detach().clone()
                )
                continue
            sh = self._shadow[k]
            if v.is_floating_point():
                sh.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                # Buffers like running counts — keep up to date.
                sh.copy_(v.detach())


    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the shadow as a plain state_dict (float32 for fp tensors)."""
        return {k: v.detach().clone() for k, v in self._shadow.items()}

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily swap EMA weights into ``model`` for evaluation.

        Restores the live weights on exit, even if the body raises.
        """
        live_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Build a state_dict cast to the model's existing dtypes/devices so
        # load_state_dict doesn't trigger spurious dtype changes.
        casted = {}
        for k, v_live in live_backup.items():
            sh = self._shadow.get(k)
            if sh is None:
                # EMA has nothing for this key (rare) — keep the live value.
                casted[k] = v_live
            else:
                casted[k] = sh.to(dtype=v_live.dtype, device=v_live.device)
        model.load_state_dict(casted, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(live_backup, strict=True)

    # --- bookkeeping --------------------------------------------------------

    def to(self, device) -> "ModelEma":
        """Move the shadow to a different device. Returns self."""
        for k in list(self._shadow.keys()):
            self._shadow[k] = self._shadow[k].to(device)
        return self

    def __len__(self) -> int:
        return len(self._shadow)
