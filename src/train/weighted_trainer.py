# src/train/weighted_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from src.utils.metrics import AccMeter


@dataclass
class WeightedTrainConfig:
    epochs: int = 1
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    device: str = "cuda"
    num_classes: int = 10
    use_amp: bool = True


def train_one_epoch_weighted(
    model: nn.Module,
    loader,
    optimizer,
    cfg: WeightedTrainConfig,
    scaler: GradScaler,
    sample_weights: torch.Tensor,
) -> Dict[str, Any]:
    """
    sample_weights: shape [N_train], on CPU ok; we gather by idx then move to device.
    loader must yield (x,y,idx).
    """
    model.train()
    meter = AccMeter(cfg.num_classes)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    running_loss = 0.0

    for x, y, idx in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        w = sample_weights[idx].to(cfg.device, non_blocking=True).float()  # [B]
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=("cuda" if cfg.device == "cuda" else "cpu"), enabled=cfg.use_amp):
            logits = model(x)
            per_sample = loss_fn(logits, y)          # [B]
            # loss = (per_sample * w).mean()
            loss = (per_sample * w).sum() / (w.sum() + 1e-12)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * x.size(0)
        meter.update(logits.detach(), y)

    return {
        "loss": running_loss / max(1, meter.total),
        "acc": meter.overall_acc(),
        "per_class_acc": meter.per_class_acc(),
    }


@torch.no_grad()
def evaluate_indexed(model: nn.Module, loader, cfg: WeightedTrainConfig) -> Dict[str, Any]:
    model.eval()
    meter = AccMeter(cfg.num_classes)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    for x, y, _idx in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        running_loss += float(loss.item()) * x.size(0)
        meter.update(logits, y)

    return {
        "loss": running_loss / max(1, meter.total),
        "acc": meter.overall_acc(),
        "per_class_acc": meter.per_class_acc(),
    }

