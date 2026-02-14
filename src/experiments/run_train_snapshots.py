# src/experiments/run_train_snapshots.py
from __future__ import annotations

import argparse, os, time
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch.amp import GradScaler

from torch.utils.data import DataLoader
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.transforms import cifar10_train_aug, cifar10_noaug
from src.data.indexed import IndexedDataset
from src.models.resnet9 import ResNet9
from src.train.trainer import TrainConfig, train_one_epoch, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--train_aug", type=int, default=1)   # 1: aug, 0: noaug
    p.add_argument("--save_epochs", type=str, default="10,11,15,16")  # csv list
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_dir = os.path.join(args.out_dir, "logs")
    ensure_dir(ckpt_dir); ensure_dir(log_dir)

    train_tf = cifar10_train_aug() if args.train_aug else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_amp)

    cfg = TrainConfig(epochs=args.epochs, device=device, num_classes=10, use_amp=use_amp,
                      lr=0.1, weight_decay=5e-4, momentum=0.9)

    model = ResNet9(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    save_epochs = sorted({int(x) for x in args.save_epochs.split(",") if x.strip()})
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
            "train_acc": float(tr["acc"]),
            "test_acc": float(te["acc"]),
            "test_per_class_acc": [float(x) for x in te["per_class_acc"]],
            "sec": float(time.time() - t0),
        }
        history.append(row)

        # best checkpoint (save full training state for reproducibility)
        if te["acc"] > best_acc:
            best_acc = float(te["acc"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if cfg.use_amp else None,
                    "best_acc": best_acc,
                    "cfg": cfg.__dict__,
                    "args": vars(args),  # records save_epochs/train_aug/data_root/etc.
                },
                os.path.join(ckpt_dir, "best.pt"),
            )

        
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"[{epoch:03d}/{cfg.epochs}] test_acc={row['test_acc']:.2f} time={row['sec']:.1f}s")

        if epoch in save_epochs:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if cfg.use_amp else None,
                    "cfg": cfg.__dict__,
                    "args": vars(args),
                },
                path,
            )
            print("Saved ckpt:", path)


    save_json(os.path.join(log_dir, "history.json"), {"history": history})
    print("Out:", args.out_dir)


if __name__ == "__main__":
    main()

