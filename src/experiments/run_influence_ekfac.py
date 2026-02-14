# src/experiments/run_influence_ekfac.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from src.influence.ekfac_influence import EKFACInfluenceConfig, compute_ekfac_influence_Ptrain


def _cifar10_noaug_transforms():
    # IMPORTANT: ideally match your baseline normalization exactly.
    # CIFAR-10 common normalization:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf


def _load_ckpt_into_model(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support multiple checkpoint formats
    if isinstance(ckpt, dict):
        # for key in ["state_dict", "model", "model_state_dict", "net"]:
        for key in ["state_dict", "model", "model_state", "model_state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            # maybe directly a state_dict-like dict
            sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Remove possible 'module.' prefix
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    if missing:
        print("[CKPT] Missing keys (first 20):", missing[:20])
    if unexpected:
        print("[CKPT] Unexpected keys (first 20):", unexpected[:20])


def build_resnet9(num_classes: int) -> nn.Module:
    """
    TODO: change this to your repo's ResNet9 builder.
    Examples:
      from src.models.resnet9 import ResNet9
      return ResNet9(num_classes=num_classes)
    """
    # ---- CHANGE THIS IMPORT TO MATCH YOUR PROJECT ----
    from src.models.resnet9 import ResNet9  # <- adjust if needed
    return ResNet9(num_classes=num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # dataset / loader
    ap.add_argument("--noaug", type=int, default=1)
    ap.add_argument("--train_bs", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)

    # EKFAC influence params
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--max_factor_batches", type=int, default=200)  # set None by -1
    ap.add_argument("--max_val_per_class", type=int, default=500)
    ap.add_argument("--mode", type=str, default="full", choices=["full", "subsample"])
    ap.add_argument("--max_train_samples", type=int, default=5000)
    ap.add_argument("--layer_filter", type=str, default="")  # "" means all layers

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- dataset (no augmentation) --------
    tf = _cifar10_noaug_transforms()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=tf)
    val_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tf)

    # IMPORTANT: shuffle must be False for correct global indexing in P_train.npy
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------- model --------
    model = build_resnet9(num_classes=args.num_classes).to(device)
    model.eval()
    _load_ckpt_into_model(model, args.ckpt, device=device)

    # -------- config --------
    cfg = EKFACInfluenceConfig(
        num_classes=args.num_classes,
        damping=args.damping,
        max_factor_batches=None if args.max_factor_batches < 0 else args.max_factor_batches,
        max_val_per_class=args.max_val_per_class,
        mode=args.mode,
        max_train_samples=args.max_train_samples,
        layer_filter=args.layer_filter if args.layer_filter != "" else None,
    )

    # -------- run EKFAC influence --------
    P, meta = compute_ekfac_influence_Ptrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )

    # save
    P_path = os.path.join(args.out_dir, "P_train.npy")
    np.save(P_path, P)

    meta_all: Dict[str, Any] = {
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "device": device,
        "train_bs": args.train_bs,
        "noaug": args.noaug,
        "cfg": asdict(cfg),
        **meta,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_all, f, indent=2)

    # quick sanity print
    print("[DONE] Saved:", P_path, "shape=", P.shape)
    print("[SANITY] per-class sum(P):", P.sum(axis=0))
    print("[SANITY] mean sum(P):", float(P.sum(axis=1).mean()))


if __name__ == "__main__":
    main()
