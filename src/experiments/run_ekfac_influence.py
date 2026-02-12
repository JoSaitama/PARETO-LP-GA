# src/experiments/run_ekfac_influence.py
from __future__ import annotations

import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.transforms import cifar10_noaug, cifar10_train_aug
from src.models.resnet9 import ResNet9
from src.influence.ekfac_influence import EKFACInfluenceConfig, compute_ekfac_influence_Ptrain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, required=True)        # baseline ckpt
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)    # train batch used for factor estimation
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--noaug", type=int, default=1)           # influence uses noaug by default
    p.add_argument("--top_pct", type=float, default=0.10)

    # EKFAC specifics
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--max_factor_batches", type=int, default=200)  # reduce if needed
    p.add_argument("--max_val_per_class", type=int, default=200)
    p.add_argument("--mode", type=str, default="subsample", choices=["full", "subsample"])
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--layer_filter", type=str, default="")  # e.g. "fc" or "layer4"; empty -> all layers
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(args.out_dir)
    top_dir = os.path.join(args.out_dir, "top_lists")
    ensure_dir(top_dir)

    train_tf = cifar10_noaug() if args.noaug else cifar10_train_aug()
    test_tf = cifar10_noaug()

    # IMPORTANT: shuffle=False for correct index alignment (see meta note)
    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = ResNet9(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    layer_filter = args.layer_filter if args.layer_filter.strip() != "" else None
    cfg = EKFACInfluenceConfig(
        num_classes=10,
        damping=args.damping,
        max_factor_batches=args.max_factor_batches if args.max_factor_batches > 0 else None,
        max_val_per_class=args.max_val_per_class,
        mode=args.mode,
        max_train_samples=args.max_train_samples if args.mode == "subsample" else None,
        seed=args.seed,
        layer_filter=layer_filter,
    )

    P_train, meta = compute_ekfac_influence_Ptrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )

    P_path = os.path.join(args.out_dir, "P_train.npy")
    np.save(P_path, P_train.numpy())
    save_json(os.path.join(args.out_dir, "meta.json"), meta)

    P = P_train.numpy()
    N, K = P.shape
    m = int(N * args.top_pct)

    for k in range(K):
        scores = P[:, k]
        beneficial = np.argsort(-scores)[:m]
        detrimental = np.argsort(scores)[:m]
        np.save(os.path.join(top_dir, f"class{k}_beneficial.npy"), beneficial)
        np.save(os.path.join(top_dir, f"class{k}_detrimental.npy"), detrimental)

    print("Saved:", P_path)
    print("Saved top lists to:", top_dir)
    print("Meta:", os.path.join(args.out_dir, "meta.json"))


if __name__ == "__main__":
    main()
