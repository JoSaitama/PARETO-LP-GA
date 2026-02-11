# src/experiments/run_delete_retrain.py
from __future__ import annotations

"""
Delete & retrain experiment for Section 5.1:
- compute top beneficial/detrimental indices per class
- delete them from training set
- retrain for a short schedule
- record per-class acc changes
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json, load_json
from src.data.transforms import cifar10_noaug, cifar10_train_aug
from src.models.resnet9 import ResNet9
from src.train.trainer import TrainConfig, train_one_epoch, evaluate


def retrain_once(
    train_ds_full,
    test_loader,
    device: str,
    removed_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    num_workers: int,
    init_state_dict=None,
):
    N = len(train_ds_full)
    removed_set = set(removed_idx.tolist())
    keep_idx = [i for i in range(N) if i not in removed_set]

    train_ds = Subset(train_ds_full, keep_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = ResNet9(num_classes=10).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    cfg = TrainConfig(
        epochs=epochs,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        device=device,
        num_classes=10,
        use_amp=(device == "cuda"),
    )

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.use_amp)

    best = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, cfg, scaler)
        te = evaluate(model, test_loader, cfg)
        scheduler.step()

        if te["acc"] > best:
            best = float(te["acc"])
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if ep in (1, epochs // 2, epochs):
            print(f"  ep {ep:02d}/{epochs} test_acc={te['acc']:.2f} best={best:.2f}")

    model.load_state_dict(best_state)
    final = evaluate(model, test_loader, cfg)
    return {
        "best_acc": float(final["acc"]),
        "per_class_acc": np.array(final["per_class_acc"], dtype=np.float32),
        "keep_size": len(keep_idx),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--inf_dir", type=str, required=True)      # contains P_train.npy + top_lists/
    p.add_argument("--base_ckpt", type=str, required=True)    # baseline best.pt
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--targets", type=str, default="all")      # "all" or "0,1,2"
    p.add_argument("--train_noaug", type=int, default=0)      # 1: no aug in retrain
    p.add_argument("--resume", type=int, default=1)
    p.add_argument("--init_from_ckpt", type=int, default=0)   # 1: init retrain weights from baseline ckpt
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)

    P_path = os.path.join(args.inf_dir, "P_train.npy")
    top_dir = os.path.join(args.inf_dir, "top_lists")
    assert os.path.exists(P_path), f"missing {P_path}"
    assert os.path.isdir(top_dir), f"missing {top_dir}"

    P = np.load(P_path)  # [N, 10]
    N, K = P.shape

    # targets
    if args.targets.strip().lower() == "all":
        targets = list(range(K))
    else:
        targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]

    modes = ["beneficial", "detrimental"]

    # data
    train_tf = cifar10_noaug() if args.train_noaug else cifar10_train_aug()
    test_tf = cifar10_noaug()

    train_ds_full = datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # baseline eval
    base_model = ResNet9(num_classes=10).to(device)
    ckpt = torch.load(args.base_ckpt, map_location=device)
    base_state = ckpt["model_state"]
    base_model.load_state_dict(base_state)
    base_model.eval()
    base_cfg = TrainConfig(epochs=1, device=device, num_classes=10, use_amp=False)
    base_eval = evaluate(base_model, test_loader, base_cfg)
    base_acc = np.array(base_eval["per_class_acc"], dtype=np.float32)
    base_overall = float(base_eval["acc"])

    # resume
    partial_path = os.path.join(args.out_dir, "results_partial.json")
    done = set()
    results = []
    if args.resume and os.path.exists(partial_path):
        obj = load_json(partial_path)
        results = obj.get("results", [])
        done = set(obj.get("done", []))
        print(f"Resume: loaded {len(done)} finished runs from {partial_path}")

    rows_total = len(targets) * len(modes)
    cum_infl = np.zeros((rows_total, K), dtype=np.float32)
    acc_change = np.zeros((rows_total, K), dtype=np.float32)
    row_names = []

    row = 0
    for t in targets:
        for mode in modes:
            key = f"t{t}_{mode}"
            row_names.append(key)

            if key in done:
                row += 1
                continue

            removed = np.load(os.path.join(top_dir, f"class{t}_{mode}.npy"))

            print(f"\n=== target={t}, mode={mode}, removed={len(removed)} ===")
            cum = P[removed].sum(axis=0).astype(np.float32)
            cum_infl[row] = cum

            labels = np.array(train_ds_full.targets)
            hist = np.bincount(labels[removed], minlength=K)
            target_ratio = float(hist[t] / max(1, hist.sum()))
            print("cum influence on target class:", float(cum[t]))
            print("removed label hist:", hist)
            print("removed target ratio:", target_ratio)

            init_state = base_state if args.init_from_ckpt else None

            out = retrain_once(
                train_ds_full=train_ds_full,
                test_loader=test_loader,
                device=device,
                removed_idx=removed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                init_state_dict=init_state,
            )

            new_acc = out["per_class_acc"]
            delta = new_acc - base_acc
            acc_change[row] = delta

            rec = {
                "key": key,
                "target": t,
                "mode": mode,
                "removed": int(len(removed)),
                "keep_size": int(out["keep_size"]),
                "baseline_overall": base_overall,
                "baseline_per_class": base_acc.tolist(),
                "overall_acc": float(out["best_acc"]),
                "per_class_acc": new_acc.tolist(),
                "delta_target": float(delta[t]),
                "removed_label_hist": hist.tolist(),
                "removed_target_ratio": target_ratio,
            }
            results.append(rec)
            done.add(key)

            # save partial after each run
            save_json(partial_path, {"done": sorted(list(done)), "results": results})

            print("overall_acc:", out["best_acc"])
            print("delta(target class):", float(delta[t]))

            row += 1

    # save arrays + final results
    np.save(os.path.join(args.out_dir, "cum_influence.npy"), cum_infl)
    np.save(os.path.join(args.out_dir, "acc_change.npy"), acc_change)
    save_json(os.path.join(args.out_dir, "results.json"), {"row_names": row_names, "results": results})
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()

