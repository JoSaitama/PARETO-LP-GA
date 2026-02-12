# src/experiments/run_pareto_di.py
from __future__ import annotations

import argparse, os
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler

from torch.utils.data import DataLoader
from torchvision import datasets

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json, load_json
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
    p.add_argument("--ckpt_e", type=str, required=True)      # epoch e ckpt (e.g., epoch_010.pt)
    p.add_argument("--P_train", type=str, required=True)     # influence matrix from epoch e model
    p.add_argument("--targets", type=str, required=True)     # e.g. "0,2"
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=1)       # DI uses aug training typically
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=0.01)
    
    # search params
    p.add_argument("--pop", type=int, default=10)            # number of alpha candidates
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

    # load influence matrix
    P = np.load(args.P_train)   # [N, K]
    evr1 = explained_var_ratio_first_pc(P)
    print("Ceiling check EVR1:", evr1)

    # data loaders (indexed)
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

    # load model at epoch e
    model_e = ResNet9(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt_e, map_location=device)
    model_e.load_state_dict(ckpt["model_state"])
    cfg = WeightedTrainConfig(epochs=1, device=device, num_classes=10, use_amp=use_amp,
                              lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # baseline eval at epoch e
    base_e = evaluate_indexed(model_e, test_loader, cfg)
    base_pc = np.array(base_e["per_class_acc"], dtype=np.float32)
    print("Epoch-e overall:", base_e["acc"])
    print("Epoch-e per-class:", base_pc)

    best = None

    for i in range(args.pop):
        alpha = sample_alpha(K=K, target_classes=targets, rng=rng)
        w_np = solve_weights_projected(
            P=P, target_classes=targets, alpha=alpha,
            w_max=args.w_max, steps=args.opt_steps, seed=int(rng.integers(1e9))
        )
        w = torch.from_numpy(w_np)  # [N]

        # train 1 weighted epoch from model_e
        model = ResNet9(num_classes=10).to(device)
        model.load_state_dict(model_e.state_dict())

        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scaler = GradScaler("cuda", enabled=cfg.use_amp)

        train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        scheduler.step()
        after = evaluate_indexed(model, test_loader, cfg)
        after_pc = np.array(after["per_class_acc"], dtype=np.float32)

        delta = after_pc - base_pc
        # DI fitness: target must improve, minimize non-target drops
        eps = ags.eps  # or 1e-2
        if np.any(delta[targets] <= eps):
        # if np.any(delta[targets] <= 0):
            fit = -1e9
        else:
            non_t = [k for k in range(K) if k not in targets]
            neg = delta[non_t][delta[non_t] < 0].sum() if len(non_t) else 0.0
            fit = float(delta[targets].mean() + neg)  # targets up, non-target negative penalizes
        
        # ===== DEBUG PRINT (new) =====
        feasible = not np.any(delta[targets] <= 0)
        non_t = [k for k in range(K) if k not in targets]
        neg = delta[non_t][delta[non_t] < 0].sum() if len(non_t) else 0.0
        
        worst_non_t = delta[non_t].min() if len(non_t) else 0.0
        
        print(
            f"[cand {i}] "
            f"feasible={feasible} "
            f"fit={fit:.4f} "
            f"delta_targets={delta[targets]} "
            f"mean_target={delta[targets].mean():.4f} "
            f"neg_sum={neg:.4f} "
            f"worst_non_target={worst_non_t:.4f}"
        )

        
        rec = {
            "cand": i,
            "alpha": alpha.tolist(),
            "fitness": fit,
            "epoch_e_overall": float(base_e["acc"]),
            "epoch_e_per_class": base_pc.tolist(),
            "epoch_e1_overall": float(after["acc"]),
            "epoch_e1_per_class": after_pc.tolist(),
            "delta_per_class": delta.tolist(),
        }

        if best is None or fit > best["fitness"]:
            best = rec
            save_json(os.path.join(args.out_dir, "best.json"), best)
            np.save(os.path.join(args.out_dir, "best_weights.npy"), w_np)
        print(f"[cand {i}] fit={fit:.4f} e1_overall={after['acc']:.2f} target_delta={delta[targets].mean():.4f}")

    out = {"ceiling_evr1": evr1, "targets": targets, "best": best}
    save_json(os.path.join(args.out_dir, "summary.json"), out)
    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()

