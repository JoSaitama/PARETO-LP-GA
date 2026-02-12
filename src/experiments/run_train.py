# src/experiments/run_train.py
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import torch
import torch.optim as optim
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.cifar10 import CIFAR10Config, get_cifar10_loaders
from src.models.resnet9 import ResNet9
from src.train.trainer import TrainConfig, train_one_epoch, evaluate
from src.experiments.parse_file import add_common_args, exp_dirs


def main():
    p = argparse.ArgumentParser()

    # 1) 公共默认参数（不想每次都写的）
    # includes: --data_root --output_root --seed --batch_size --num_workers --epochs --aug
    add_common_args(p)

    # 2) 你真正会经常调的训练超参（这里给默认值，但可随时覆盖）
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    # 3) 输出目录：不再 required；不传就自动生成
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional. If empty, will auto-generate under output_root.",
    )

    # 4) 额外：打印频率（可选）
    p.add_argument("--print_every", type=int, default=10)

    args = p.parse_args()

    # ===== paths =====
    # 如果没传 out_dir，就自动用标准命名
    dirs = exp_dirs(args.output_root, "resnet9_cifar10", args.seed)
    out_dir = args.out_dir or dirs["baseline"]

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    # ===== device =====
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    # ===== data =====
    data_cfg = CIFAR10Config(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        aug=args.aug,  # "train_aug" or "noaug" (from CIFAR10Config)
    )
    train_loader, test_loader = get_cifar10_loaders(data_cfg)

    # ===== train cfg =====
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
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
    scaler = GradScaler("cuda", enabled=cfg.use_amp)
    # scaler = GradScaler(enabled=cfg.use_amp)

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
            "aug": args.aug,
            "device": device,
        }
        history.append(row)

        # best checkpoint
        if te["acc"] > best_acc:
            best_acc = float(te["acc"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg.__dict__,
                    "data_cfg": data_cfg.__dict__,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == cfg.epochs:
            print(
                f"[{epoch:03d}/{cfg.epochs}] lr={row['lr']:.5f} "
                f"train_acc={row['train_acc']:.2f} test_acc={row['test_acc']:.2f} "
                f"best={best_acc:.2f} time={row['sec']:.1f}s"
            )

    save_json(os.path.join(log_dir, "history.json"), {"history": history})

    print("\nDONE. best_acc =", best_acc)
    print("Out dir:", out_dir)
    print("Best checkpoint:", os.path.join(ckpt_dir, "best.pt"))
    print("History:", os.path.join(log_dir, "history.json"))
    print("Used aug:", args.aug, "| epochs:", args.epochs, "| lr:", args.lr)


if __name__ == "__main__":
    main()
