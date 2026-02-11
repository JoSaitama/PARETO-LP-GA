# src/experiments/run_influence.py
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.transforms import cifar10_noaug, cifar10_train_aug
from src.models.resnet9 import ResNet9
from src.influence.cw_influence import HeadOnlyInfluenceConfig, compute_cw_influence_head_only
from src.experiments.parse_file import add_common_args, exp_dirs, ckpt_path


def main():
    p = argparse.ArgumentParser()

    # 公共默认参数：data_root/output_root/seed/batch_size/num_workers/epochs/aug...
    add_common_args(p)

    # influence 专属参数
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Optional. If empty, will use baseline best.pt under output_root.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional. If empty, will auto-generate under output_root.",
    )

    # influence 是否禁用增强（默认禁用，更贴你要的 no-aug influence）
    p.add_argument("--noaug", type=int, default=1, help="1: use no-aug for influence dataset")

    # top list
    p.add_argument("--top_pct", type=float, default=0.10)

    # head-only influence params
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--max_val_per_class", type=int, default=1000)

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== default paths =====
    dirs = exp_dirs(args.output_root, "resnet9_cifar10", args.seed)

    baseline_dir = dirs["baseline"]
    influence_dir_default = os.path.join(
        args.output_root,
        f"influence_resnet9_cifar10_seed{args.seed}_headonly_" + ("noaug" if args.noaug else "aug"),
    )

    out_dir = args.out_dir or influence_dir_default

    ckpt = args.ckpt or ckpt_path(baseline_dir)

    ensure_dir(out_dir)
    top_dir = os.path.join(out_dir, "top_lists")
    ensure_dir(top_dir)

    # ===== transforms =====
    train_tf = cifar10_noaug() if args.noaug else cifar10_train_aug()
    test_tf = cifar10_noaug()

    # ===== datasets/loaders =====
    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # ===== model =====
    model = ResNet9(num_classes=10).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_obj["model_state"])
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

    # ===== save =====
    P_path = os.path.join(out_dir, "P_train.npy")
    meta_path = os.path.join(out_dir, "meta.json")

    np.save(P_path, P_train.detach().cpu().numpy())
    save_json(meta_path, meta)

    P = np.load(P_path)   # [N, 10]
    N, K = P.shape
    m = int(N * args.top_pct)

    for k in range(K):
        scores = P[:, k].astype(np.float64)

        # beneficial = top (largest), detrimental = bottom (smallest)
        beneficial = np.argsort(-scores)[:m]
        detrimental = np.argsort(scores)[:m]

        np.save(os.path.join(top_dir, f"class{k}_beneficial.npy"), beneficial)
        np.save(os.path.join(top_dir, f"class{k}_detrimental.npy"), detrimental)

    print("\n=== Influence DONE ===")
    print("device:", device)
    print("ckpt:", ckpt)
    print("data_root:", args.data_root)
    print("noaug:", bool(args.noaug))
    print("out_dir:", out_dir)
    print("Saved:", P_path)
    print("Saved:", meta_path)
    print("Saved top lists to:", top_dir)


if __name__ == "__main__":
    main()
