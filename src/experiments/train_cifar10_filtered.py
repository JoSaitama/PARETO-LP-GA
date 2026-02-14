import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 你需要把下面两个 import 换成你项目里实际的
from src.models.resnet9 import ResNet9
from src.train.train_one_epoch import train_one_epoch   # 如果你没有这个路径，就改成你实际位置
from src.train.eval import evaluate                     # 同上

def cifar10_noaug_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def load_remove_indices(remove_idx_npy: str | None):
    if remove_idx_npy is None:
        return None
    idx = np.load(remove_idx_npy).astype(np.int64)
    idx = np.unique(idx)
    return set(idx.tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train_bs", type=int, default=256)
    ap.add_argument("--eval_bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--remove_idx_npy", type=str, default=None)
    ap.add_argument("--noaug", type=int, default=1)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = cifar10_noaug_transform() if args.noaug == 1 else cifar10_noaug_transform()  # 你目前只做 noaug
    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=tf)
    test_ds  = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tf)

    remove_set = load_remove_indices(args.remove_idx_npy)

    if remove_set is None:
        kept_indices = list(range(len(train_ds)))
    else:
        kept_indices = [i for i in range(len(train_ds)) if i not in remove_set]

    train_subset = Subset(train_ds, kept_indices)

    train_loader = DataLoader(train_subset, batch_size=args.train_bs, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.eval_bs, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    model = ResNet9(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 你 baseline 用什么 scheduler/amp，就照搬。这里先给最简版：
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    best_acc = -1.0
    history = []

    for ep in range(1, args.epochs + 1):
        # 训练
        tr = train_one_epoch(model, train_loader, optimizer, device=device)  # 你按自己的签名改
        # 测试
        te = evaluate(model, test_loader, device=device)                     # 你按自己的签名改

        scheduler.step()

        row = {"epoch": ep, **tr, **te, "kept_train_n": len(kept_indices)}
        history.append(row)
        print(row)

        if te.get("acc", -1.0) > best_acc:
            best_acc = te["acc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best_acc": best_acc}, os.path.join(args.out_dir, "best.pt"))

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("[DONE] out_dir:", args.out_dir, "best_acc:", best_acc)

if __name__ == "__main__":
    main()
