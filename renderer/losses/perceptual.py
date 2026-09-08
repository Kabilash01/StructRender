"""Perceptual reconstruction loss — `L_rec = w_l1 * L1 + w_perc * LPIPS`.

KB §4 *Training Loss Module* and Decision 4 list reconstruction **and
perceptual** as core losses; only the adversarial term is optional. M4 shipped
with L1 alone, which was adequate while the teacher target was nearly untextured
— and stopped being adequate the moment the reference style carried real
brushwork.

Measured on the enriched scene (KB Decision 41): an L1-only baseline retains
**0.751** of the target's high-frequency energy against a 0.9 floor, i.e. it is
visibly blurry. That is L1's textbook failure mode — it regresses to the blurry
conditional mean of the plausible outputs — and it only became visible once the
target had high-frequency content to lose.

The perceptual term is the same LPIPS(AlexNet) implementation the evaluation
reports use. Sharing it means the training objective and the reported metric
agree about what "looks right" means; it also means LPIPS is **not** an
independent check on this model, which is why the sharpness ratio and the
structural gates stay in the acceptance set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PerceptualReconstructionLoss(nn.Module):
    """L1 + LPIPS, both on `[0,1]` RGB tensors shaped `(B, 3, H, W)`.

    When `lpips` is unavailable this degrades to pure L1 and sets
    `used_perceptual = False`. That flag is the *only* intended use of the
    degraded mode: **both callers in `train_baseline.py` raise on it**, so no
    training run reaches a ladder number under a quietly different objective
    (KB Decision 41/42 — the confound Decision 35 exists to prevent). The
    degraded path is kept so importing and unit-testing this module works on a
    clean checkout, not so training can fall back. Do not "fix" those raises.
    """

    def __init__(self, w_l1: float = 1.0, w_perceptual: float = 0.0, net: str = "alex"):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_perceptual = float(w_perceptual)
        self._lpips = None
        self.used_perceptual = False
        self.net = net
        if self.w_perceptual > 0.0:
            # ImportError ONLY. A broader catch would swallow a genuine failure
            # inside LPIPS (bad weights, CUDA fault) and silently report it as
            # "unavailable", turning a bug into a quietly different objective —
            # the exact confound KB Decision 35 warns about.
            try:
                import lpips as _lpips
            except ImportError:
                self._lpips = None
                self.used_perceptual = False
            else:
                self._lpips = _lpips.LPIPS(net=net, verbose=False)
                for p in self._lpips.parameters():      # frozen feature extractor
                    p.requires_grad_(False)
                self._lpips.eval()
                self.used_perceptual = True

    def describe(self) -> str:
        """Human-readable objective, published into every training report so an
        artifact always names the loss it was produced under."""
        if self.used_perceptual:
            return f"{self.w_l1:g}*L1 + {self.w_perceptual:g}*LPIPS({self.net})"
        return f"{self.w_l1:g}*L1"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        total = self.w_l1 * l1
        parts = {"l1": float(l1.detach())}
        if self._lpips is not None:
            # LPIPS expects [-1, 1]
            perc = self._lpips(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()
            total = total + self.w_perceptual * perc
            parts["perceptual"] = float(perc.detach())
        parts["total"] = float(total.detach())
        return total, parts
