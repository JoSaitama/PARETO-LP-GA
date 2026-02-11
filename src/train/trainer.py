from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.utils.metrics import AccMeter


@dataclass
class TrainConfig:
    epochs: int = 120
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    device: str = "cuda"
    num_classes: int = 10
    use_amp: bool = True


def train_one_epoch(model: nn.Module, loader, optimizer, cfg: TrainConfig, scaler: GradScaler) -> Dict[str, Any]:
    model.train()
    meter = AccMeter(cfg.num_classes)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

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
def evaluate(model: nn.Module, loader, cfg: TrainConfig) -> Dict[str, Any]:
    model.eval()
    meter = AccMeter(cfg.num_classes)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    for x, y in loader:
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
