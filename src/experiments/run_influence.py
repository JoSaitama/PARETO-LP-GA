# src/experiments/run_influence.py
from __future__ import annotations

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.transforms import cifar10_noaug, cifar10_train_aug
from src.models.resnet9 import ResNet9
from src.influence.cw_influence import HeadOnlyInfluenceConfig, compute_cw_influence_head_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, required=True)        # .../checkpoints/best.pt
    p.add_argument("--data_root", type=str, required=True)   # .../cifar10_data
    p.add_argument("--out_dir", type=str, required=True)     # .../influence_xxx
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--influence_noaug", type=int, default=1) # 1: no aug for influence
    p.add_argument("--top_pct", type=float, default=0.10)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--max_val_per_class", type=int, default=1000)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(args.out_dir)
    top_dir = os.path.join(args.out_dir, "top_lists")
    ensure_dir(top_dir)

    train_tf = cifar10_noaug() if args.influence_noaug else cifar10_train_aug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=test_tf)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = ResNet9(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg_inf = HeadOnlyInfluenceConfig(
        num_classes=10,
        ridge=args.ridge,
        max_train_samples=None,
        max_val_samples_per_class=args.max_val_per_class,
    )

    P_train, meta = compute_cw_influence_head_only(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        cfg=cfg_inf,
    )

    P_path = os.path.join(args.out_dir, "P_train.npy")
    meta_path = os.path.join(args.out_dir, "meta.json")
    np.save(P_path, P_train.detach().cpu().numpy())
    save_json(meta_path, meta)

    # top lists
    P = np.load(P_path)   # [N, 10]
    N, K = P.shape
    m = int(N * args.top_pct)

    for k in range(K):
        scores = P[:, k]

        # IMPORTANT: beneficial = top (largest), detrimental = bottom (smallest)
        beneficial = np.argsort(-scores)[:m]
        detrimental = np.argsort(scores)[:m]

        np.save(os.path.join(top_dir, f"class{k}_beneficial.npy"), beneficial)
        np.save(os.path.join(top_dir, f"class{k}_detrimental.npy"), detrimental)

    print("Saved:", P_path)
    print("Saved:", meta_path)
    print("Saved top lists to:", top_dir)


if __name__ == "__main__":
    main()
