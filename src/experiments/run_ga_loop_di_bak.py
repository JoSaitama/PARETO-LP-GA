# src/experiments/run_ga_loop_di.py
from __future__ import annotations

import argparse
import os
import json
from typing import Any, Dict, List, Tuple

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
from src.train.weighted_trainer import (
    WeightedTrainConfig,
    train_one_epoch_weighted,
    evaluate_indexed,
)
from src.pareto.ceiling_pca import explained_var_ratio_first_pc
from src.pareto.ga_search import sample_alpha
from src.pareto.weight_opt import solve_weights_projected


def _project_alpha(alpha: np.ndarray, mode: str = "clip") -> np.ndarray:
    """
    clip     : clip each alpha_k to [0,1]  (paper-consistent threshold range)
    none     : leave alpha unchanged
    simplex  : nonnegative & sum=1 (ablation only)
    """
    a = alpha.astype(np.float32)
    if mode == "clip":
        return np.clip(a, 0.0, 1.0)
    if mode == "none":
        return a
    if mode == "simplex":
        a = np.maximum(a, 0.0)
        s = float(a.sum())
        if s <= 1e-12:
            return np.ones_like(a, dtype=np.float32) / float(a.size)
        return (a / s).astype(np.float32)
    raise ValueError(f"Unknown alpha_project mode: {mode}")


def _tournament_select(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    n = int(fits.shape[0])
    k = max(1, min(int(k), n))
    idx = rng.integers(0, n, size=k)
    best = int(idx[0])
    best_fit = float(fits[best])
    for j in idx[1:]:
        jj = int(j)
        if float(fits[jj]) > best_fit:
            best_fit = float(fits[jj])
            best = jj
    return best


def _crossover(
    rng: np.random.Generator,
    a1: np.ndarray,
    a2: np.ndarray,
    cx_beta: float,
    alpha_project: str,
) -> np.ndarray:
    beta = float(cx_beta)
    beta = min(max(beta, 0.5), 1.0)
    u = float(rng.uniform(1.0 - beta, beta))
    child = u * a1 + (1.0 - u) * a2
    return _project_alpha(child, alpha_project)


def _mutate(
    rng: np.random.Generator,
    a: np.ndarray,
    mut_rate: float,
    mut_sigma: float,
    alpha_project: str,
) -> np.ndarray:
    child = a.astype(np.float32).copy()
    K = int(child.size)
    m = int(np.ceil(float(mut_rate) * K))
    m = max(1, min(K, m))
    idx = rng.choice(K, size=m, replace=False)
    child[idx] += rng.normal(loc=0.0, scale=float(mut_sigma), size=m).astype(np.float32)
    return _project_alpha(child, alpha_project)


def pick_di_targets_from_ckpt(
    model_e: torch.nn.Module,
    test_loader: DataLoader,
    cfg: WeightedTrainConfig,
    top_n: int = 2,
) -> Tuple[List[int], np.ndarray]:
    res_e = evaluate_indexed(model_e, test_loader, cfg)
    acc_e = np.array(res_e["per_class_acc"], dtype=np.float32)
    order = np.argsort(acc_e)  # lowest first
    top_n = int(max(1, min(top_n, acc_e.size)))
    targets = [int(i) for i in order[:top_n]]
    return targets, acc_e


def _make_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_e", type=str, required=True)
    p.add_argument("--ckpt_base_e1", type=str, default="")  # optional, for logging only
    p.add_argument("--P_train", type=str, required=True)

    p.add_argument("--targets", type=str, default="")
    p.add_argument("--auto_targets", type=int, default=0)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    # NOTE: paper uses lr=1e-4, wd=1e-4 for PARETO-LP-GA stage (Table 2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--eps", type=float, default=0.0)

    p.add_argument("--w_max", type=float, default=10.0)
    p.add_argument("--opt_steps", type=int, default=800)

    # GA
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--tourn_k", type=int, default=3)
    p.add_argument("--cx_beta", type=float, default=0.7)
    p.add_argument("--mut_rate", type=float, default=0.25)
    p.add_argument("--mut_sigma", type=float, default=0.25)
    p.add_argument("--init_from_best_json", type=str, default="")

    p.add_argument(
        "--alpha_project",
        type=str,
        default="clip",
        choices=["clip", "none", "simplex"],
    )

    args = p.parse_args()

    set_seed(args.seed)
    _make_deterministic()
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ensure_dir(args.out_dir)

    K = 10
    P = np.load(args.P_train)
    if P.ndim != 2 or P.shape[1] != K:
        raise ValueError(f"P_train shape expected (N,{K}), got {P.shape}.")

    S = P.sum(axis=0)
    print("[P_check] sum(P) per class:", S)
    print("[P_check] mean sum(P):", float(S.mean()))
    if float(S.mean()) < 0.0:
        print("[P_check] Flipping P <- -P to match LP direction.")
        P = -P

    evr1 = explained_var_ratio_first_pc(P)
    print("Ceiling check EVR1:", evr1)

    train_tf = cifar10_train_aug() if args.train_aug else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_idx_ds = IndexedDataset(train_ds)
    test_idx_ds = IndexedDataset(test_ds)

    train_loader = DataLoader(
        train_idx_ds,
        batch_size=args.batch_size,
        shuffle=False,   # keep fixed order for GA stability
        num_workers=args.num_workers,
        pin_memory=bool(use_amp),
    )
    test_loader = DataLoader(
        test_idx_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(use_amp),
    )

    cfg = WeightedTrainConfig(
        epochs=1,
        device=device,
        num_classes=K,
        use_amp=use_amp,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    ckpt_e = torch.load(args.ckpt_e, map_location=device)
    model_e = ResNet9(num_classes=K).to(device)
    model_e.load_state_dict(ckpt_e["model_state"])

    if int(args.auto_targets) > 0:
        targets, acc_e = pick_di_targets_from_ckpt(model_e, test_loader, cfg, top_n=int(args.auto_targets))
        print("\n[AutoTargets] Picked DI targets as LOW-ACC classes at epoch-e:", targets)
        print("[AutoTargets] epoch-e per-class:", acc_e)
    else:
        targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
        if len(targets) == 0:
            raise ValueError("targets is empty. Use --targets like '2,3', or set --auto_targets > 0.")

    base_e = evaluate_indexed(model_e, test_loader, cfg)
    base_pc = np.array(base_e["per_class_acc"], dtype=np.float32)
    print("Epoch-e overall:", base_e["acc"])
    print("Epoch-e per-class:", base_pc)

    if args.ckpt_base_e1:
        base_e1 = torch.load(args.ckpt_base_e1, map_location=device)
        m1 = ResNet9(num_classes=K).to(device)
        m1.load_state_dict(base_e1["model_state"])
        res1 = evaluate_indexed(m1, test_loader, cfg)
        print("Baseline epoch-(e+1) overall:", res1["acc"])
        print("Baseline epoch-(e+1) per-class:", res1["per_class_acc"])

    t_arr = np.array(targets, dtype=np.int64)

    def evaluate_alpha(alpha: np.ndarray):
        alpha = _project_alpha(alpha, args.alpha_project)

        w_np = solve_weights_projected(
            P=P,
            target_classes=targets,
            alpha=alpha,
            w_max=args.w_max,
            steps=args.opt_steps,
            seed=int(args.seed),
        )
        w_np = np.asarray(w_np, dtype=np.float32).reshape(-1)

        print("for debug w_stats:",
              float(w_np.min()), float(w_np.max()), float(w_np.mean()),
              "nz%", float((w_np > 0).mean()),
              "high%", float((w_np >= float(args.w_max) - 1e-6).mean()))

        # TRAINONEEPOCH from ckpt_e (restore train state if available)
        model = ResNet9(num_classes=K).to(device)
        model.load_state_dict(ckpt_e["model_state"])

        optimizer = optim.SGD(
            model.parameters(),
            lr=float(ckpt_e.get("cfg", {}).get("lr", cfg.lr)),
            momentum=float(ckpt_e.get("cfg", {}).get("momentum", cfg.momentum)),
            weight_decay=float(ckpt_e.get("cfg", {}).get("weight_decay", cfg.weight_decay)),
        )
        if "optimizer_state" in ckpt_e and ckpt_e["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt_e["optimizer_state"])

        tmax = int(ckpt_e.get("cfg", {}).get("epochs", 1))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, tmax))
        if "scheduler_state" in ckpt_e and ckpt_e["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt_e["scheduler_state"])

        scaler = GradScaler("cuda", enabled=cfg.use_amp)
        if cfg.use_amp and ("scaler_state" in ckpt_e) and (ckpt_e["scaler_state"] is not None):
            scaler.load_state_dict(ckpt_e["scaler_state"])

        w = torch.from_numpy(w_np).to(device)
        train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        scheduler.step()

        after = evaluate_indexed(model, test_loader, cfg)
        after_pc = np.array(after["per_class_acc"], dtype=np.float32)

        # DI: delta is w-epoch-(e+1) minus epoch-e
        delta = after_pc - base_pc

        eps = float(args.eps)
        feasible = bool(np.all(delta[t_arr] > eps))

        target_set = set(targets)
        non_t = [k for k in range(K) if k not in target_set]

        # paper Line-7: (1/|non_t|) sum_{k in non_t} 1[delta_k<0] * delta_k
        neg = (delta[non_t] * (delta[non_t] < 0)).astype(np.float32) if len(non_t) else np.array([], dtype=np.float32)
        if not feasible:
            fit = -1e9
        else:
            fit = float(neg.mean()) if len(non_t) else 0.0

        rec: Dict[str, Any] = {
            "alpha": alpha.tolist(),
            "fitness": float(fit),
            "feasible": bool(feasible),
            "epoch_e_overall": float(base_e["acc"]),
            "epoch_e_per_class": base_pc.tolist(),
            "new_e1_overall": float(after["acc"]),
            "new_e1_per_class": after_pc.tolist(),
            "delta_vs_epoch_e": delta.tolist(),
            "delta_targets": delta[targets].tolist(),
            "mean_target": float(delta[targets].mean()),
        }
        return float(fit), bool(feasible), rec, w_np

    # init population
    pop: List[np.ndarray] = []
    if args.init_from_best_json:
        try:
            with open(args.init_from_best_json, "r") as f:
                bj = json.load(f)
            if isinstance(bj, dict) and "alpha" in bj:
                a0 = np.array(bj["alpha"], dtype=np.float32)
                if a0.size == K:
                    pop.append(_project_alpha(a0, args.alpha_project))
                    print("Seeded GA with alpha from:", args.init_from_best_json)
        except Exception as e:
            print("WARN: failed to load init_from_best_json:", e)

    while len(pop) < int(args.pop):
        pop.append(_project_alpha(sample_alpha(K, targets, rng), args.alpha_project))

    best_fit = -1e18
    best_rec: Dict[str, Any] | None = None
    best_w: np.ndarray | None = None

    for g in range(1, int(args.gens) + 1):
        print(f"\n=== GEN {g:03d}/{int(args.gens):03d} ===")

        fits = np.zeros(len(pop), dtype=np.float32)
        feas = np.zeros(len(pop), dtype=bool)
        recs: List[Dict[str, Any]] = []

        for i, a in enumerate(pop):
            fit, ok, rec, w = evaluate_alpha(a)
            fits[i] = float(fit)
            feas[i] = bool(ok)
            recs.append(rec)
            print(f"[gen {g} cand {i}] feasible={ok} fit={fit:.4f} delta_targets={rec['delta_targets']} mean_target={rec['mean_target']:.4f}")

        elite_n = int(min(args.elite, len(pop)))
        elite_idx = np.argsort(fits)[-elite_n:][::-1]

        if float(fits[elite_idx[0]]) > float(best_fit):
            best_fit = float(fits[elite_idx[0]])
            best_rec = recs[int(elite_idx[0])]
            # re-evaluate to fetch weights for the best alpha (cheap enough)
            _, _, _, best_w = evaluate_alpha(np.array(best_rec["alpha"], dtype=np.float32))
            save_json(best_rec, os.path.join(args.out_dir, "best.json"))
            np.save(os.path.join(args.out_dir, "best_w.npy"), best_w)
            print("Updated GLOBAL BEST ->", os.path.join(args.out_dir, "best.json"))

        # Next generation
        new_pop: List[np.ndarray] = [pop[int(i)].copy() for i in elite_idx]

        while len(new_pop) < int(args.pop):
            p1 = pop[_tournament_select(rng, fits, args.tourn_k)]
            p2 = pop[_tournament_select(rng, fits, args.tourn_k)]
            child = _crossover(rng, p1, p2, args.cx_beta, args.alpha_project)
            child = _mutate(rng, child, args.mut_rate, args.mut_sigma, args.alpha_project)
            new_pop.append(child)

        pop = new_pop

    print("\nDONE. Best fitness:", best_fit)
    if best_rec is not None:
        print("Best record saved at:", os.path.join(args.out_dir, "best.json"))


if __name__ == "__main__":
    main()
