# src/experiments/run_train.py
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.cifar10 import CIFAR10Config, get_cifar10_loaders
from src.models.resnet9 import ResNet9
from src.train.trainer import TrainConfig, train_one_epoch, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_root", type=str, required=True)   # e.g. /content/drive/MyDrive/cifar10_data
    p.add_argument("--out_dir", type=str, required=True)     # e.g. /content/drive/MyDrive/pareto_influence_outputs/baseline_...
    p.add_argument("--aug", type=str, default="train_aug", choices=["train_aug", "noaug"])
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_dir = os.path.join(args.out_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    data_cfg = CIFAR10Config(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        aug=args.aug, 
    )
    train_loader, test_loader = get_cifar10_loaders(data_cfg)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        device=device,
        num_classes=10,
        use_amp=use_amp,
    )

    model = ResNet9(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.use_amp)

    best_acc = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, cfg, scaler)
        te = evaluate(model, test_loader, cfg)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "test_loss": float(te["loss"]),
            "test_acc": float(te["acc"]),
            "test_per_class_acc": [float(x) for x in te["per_class_acc"]],
            "sec": float(time.time() - t0),
        }
        history.append(row)

        if te["acc"] > best_acc:
            best_acc = float(te["acc"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg.__dict__,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"[{epoch:03d}/{cfg.epochs}] lr={row['lr']:.5f} "
                f"train_acc={row['train_acc']:.2f} test_acc={row['test_acc']:.2f} "
                f"best={best_acc:.2f} time={row['sec']:.1f}s"
            )

    save_json(os.path.join(log_dir, "history.json"), {"history": history})
    print("DONE. best_acc =", best_acc)
    print("Best checkpoint:", os.path.join(ckpt_dir, "best.pt"))
    print("History:", os.path.join(log_dir, "history.json"))


if __name__ == "__main__":
    main()

