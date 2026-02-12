# src/experiments/run_pareto_cc.py
from __future__ import annotations

import argparse, os
import numpy as np
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
from src.train.weighted_trainer import WeightedTrainConfig, train_one_epoch_weighted, evaluate_indexed
from src.pareto.ceiling_pca import explained_var_ratio_first_pc
from src.pareto.ga_search import sample_alpha
from src.pareto.weight_opt import solve_weights_projected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_e", type=str, required=True)          # epoch e ckpt, e.g. epoch_015.pt
    p.add_argument("--ckpt_orig_e1", type=str, required=True)    # original epoch e+1 ckpt, e.g. epoch_016.pt
    p.add_argument("--P_train", type=str, required=True)         # influence matrix from epoch e model
    p.add_argument("--targets", type=str, required=True)         # e.g. "5,7,9"
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--pop", type=int, default=10)
    p.add_argument("--w_max", type=float, default=5.0)
    p.add_argument("--opt_steps", type=int, default=400)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ensure_dir(args.out_dir)

    targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
    K = 10

    P = np.load(args.P_train)
    evr1 = explained_var_ratio_first_pc(P)
    print("Ceiling check EVR1:", evr1)

    train_tf = cifar10_train_aug() if args.train_aug else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_idx_ds = IndexedDataset(train_ds)
    test_idx_ds = IndexedDataset(test_ds)

    train_loader = DataLoader(train_idx_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp)
    test_loader = DataLoader(test_idx_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_amp)

    cfg = WeightedTrainConfig(epochs=1, device=device, num_classes=10, use_amp=use_amp,
                              lr=0.1, weight_decay=5e-4, momentum=0.9)

    # load epoch e and orig epoch e+1
    model_e = ResNet9(num_classes=10).to(device)
    model_orig = ResNet9(num_classes=10).to(device)
    ckpt_e = torch.load(args.ckpt_e, map_location=device)
    ckpt_o = torch.load(args.ckpt_orig_e1, map_location=device)
    model_e.load_state_dict(ckpt_e["model_state"])
    model_orig.load_state_dict(ckpt_o["model_state"])

    orig = evaluate_indexed(model_orig, test_loader, cfg)
    orig_pc = np.array(orig["per_class_acc"], dtype=np.float32)
    print("Orig e+1 overall:", orig["acc"])
    print("Orig e+1 per-class:", orig_pc)

    best = None

    for i in range(args.pop):
        alpha = sample_alpha(K=K, target_classes=targets, rng=rng)
        w_np = solve_weights_projected(
            P=P, target_classes=targets, alpha=alpha,
            w_max=args.w_max, steps=args.opt_steps, seed=int(rng.integers(1e9))
        )
        w = torch.from_numpy(w_np)

        # train 1 weighted epoch from epoch-e model
        model = ResNet9(num_classes=10).to(device)
        model.load_state_dict(model_e.state_dict())

        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scaler = GradScaler("cuda", enabled=cfg.use_amp)

        train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        scheduler.step()

        after = evaluate_indexed(model, test_loader, cfg)
        after_pc = np.array(after["per_class_acc"], dtype=np.float32)

        # CC delta is vs ORIGINAL epoch e+1
        delta = after_pc - orig_pc

        # CC fitness: targets must improve relative to orig e+1
        if np.any(delta[targets] <= 0):
            fit = -1e9
        else:
            non_t = [k for k in range(K) if k not in targets]
            neg = delta[non_t][delta[non_t] < 0].sum() if len(non_t) else 0.0
            fit = float(delta[targets].mean() + neg)

        rec = {
            "cand": i,
            "alpha": alpha.tolist(),
            "fitness": fit,
            "orig_e1_overall": float(orig["acc"]),
            "orig_e1_per_class": orig_pc.tolist(),
            "new_e1_overall": float(after["acc"]),
            "new_e1_per_class": after_pc.tolist(),
            "delta_vs_orig_per_class": delta.tolist(),
        }

        if best is None or fit > best["fitness"]:
            best = rec
            save_json(os.path.join(args.out_dir, "best.json"), best)
            np.save(os.path.join(args.out_dir, "best_weights.npy"), w_np)

        print(f"[cand {i}] fit={fit:.4f} new_overall={after['acc']:.2f} target_delta={delta[targets].mean():.4f}")

    out = {"ceiling_evr1": evr1, "targets": targets, "best": best}
    save_json(os.path.join(args.out_dir, "summary.json"), out)
    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()

