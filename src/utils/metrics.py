
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class AccMeter:
    """
    Track overall accuracy + per-class accuracy.
    """

    num_classes: int
    correct: int = 0
    total: int = 0

    def __post_init__(self):
        self.class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self.class_total = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update accuracy stats.

        logits: [B, C]
        y: [B]
        """

        preds = torch.argmax(logits, dim=1)

        self.correct += int((preds == y).sum().item())
        self.total += int(y.numel())

        for c in range(self.num_classes):
            mask = (y == c)
            if mask.any():
                self.class_correct[c] += int((preds[mask] == y[mask]).sum().item())
                self.class_total[c] += int(mask.sum().item())

    def overall_acc(self) -> float:
        return 100.0 * self.correct / max(1, self.total)

    def per_class_acc(self) -> list[float]:
        acc = []
        for c in range(self.num_classes):
            denom = int(self.class_total[c].item())
            acc.append(
                100.0 * int(self.class_correct[c].item()) / max(1, denom)
            )
        return acc
