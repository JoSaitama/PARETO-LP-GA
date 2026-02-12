# src/experiments/run_ga_loop_cc.py
from __future__ import annotations

import argparse, os
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
from src.train.weighted_trainer import WeightedTrainConfig, train_one_epoch_weighted, evaluate_indexed
from src.pareto.ceiling_pca import explained_var_ratio_first_pc
from src.pareto.ga_search import sample_alpha
from src.pareto.weight_opt import solve_weights_projected


def _normalize_alpha(alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project alpha to simplex-like: nonnegative and sum=1."""
    a = np.maximum(alpha, 0.0)
    s = float(a.sum())
    if s < eps:
        # fallback to uniform
        a[:] = 1.0 / a.size
        return a
    return a / s


def _tournament_select(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    """Return index of winner among k random indices."""
    n = fits.shape[0]
    idx = rng.integers(0, n, size=k)
    best = idx[0]
    best_fit = fits[best]
    for j in idx[1:]:
        if fits[j] > best_fit:
            best_fit = fits[j]
            best = j
    return int(best)


def _crossover(rng: np.random.Generator, a1: np.ndarray, a2: np.ndarray, cx_beta: float) -> np.ndarray:
    """Blend crossover: child = u*a1 + (1-u)*a2, u ~ Uniform(1-beta, beta) if beta>=0.5."""
    # Make cx_beta in [0.5, 1.0] meaningful; if user passes 0.5 -> fixed 0.5
    beta = float(cx_beta)
    lo = 1.0 - beta
    hi = beta
    u = float(rng.uniform(lo, hi))
    child = u * a1 + (1.0 - u) * a2
    return _normalize_alpha(child)


def _mutate(rng: np.random.Generator, a: np.ndarray, mut_rate: float, mut_sigma: float) -> np.ndarray:
    """Gaussian mutation on a subset of dims, then normalize."""
    child = a.copy()
    K = child.size
    m = int(np.ceil(mut_rate * K))
    m = max(1, min(K, m))
    idx = rng.choice(K, size=m, replace=False)
    child[idx] += rng.normal(loc=0.0, scale=mut_sigma, size=m).astype(child.dtype)
    return _normalize_alpha(child)


def main():
    p = argparse.ArgumentParser()

    # ===== same core args as run_pareto_cc.py =====
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_e", type=str, required=True)          # epoch e ckpt, e.g. epoch_020.pt
    p.add_argument("--ckpt_orig_e1", type=str, required=True)    # original epoch e+1 ckpt, e.g. epoch_021.pt
    p.add_argument("--P_train", type=str, required=True)         # influence matrix from epoch e model
    p.add_argument("--targets", type=str, required=True)         # e.g. "3" or "2,3"
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=0.01)

    p.add_argument("--w_max", type=float, default=10.0)
    p.add_argument("--opt_steps", type=int, default=400)
    p.add_argument("--lambda_shortfall", type=float, default=50.0)

    # ===== GA params =====
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--tourn_k", type=int, default=3)
    p.add_argument("--cx_beta", type=float, default=0.5)     # 0.5 => average; 0.7 => more parent-biased
    p.add_argument("--mut_rate", type=float, default=0.3)    # fraction of alpha dims to perturb (K=10)
    p.add_argument("--mut_sigma", type=float, default=0.15)  # alpha noise scale
    p.add_argument("--init_from_best_json", type=str, default="")  # optional: seed population with alpha from a CC run

    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ensure_dir(args.out_dir)

    targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
    K = 10

    # ===== load influence matrix & ceiling check =====
    P = np.load(args.P_train)
    evr1 = explained_var_ratio_first_pc(P)
    print("Ceiling check EVR1:", evr1)

    # ===== data =====
    train_tf = cifar10_train_aug() if args.train_aug else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_idx_ds = IndexedDataset(train_ds)
    test_idx_ds = IndexedDataset(test_ds)

    train_loader = DataLoader(
        train_idx_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_amp
    )
    test_loader = DataLoader(
        test_idx_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_amp
    )

    cfg = WeightedTrainConfig(
        epochs=1, device=device, num_classes=10, use_amp=use_amp,
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
    )

    # ===== load epoch e and orig epoch e+1 =====
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

    t_arr = np.array(targets, dtype=int)

    def evaluate_alpha(alpha: np.ndarray, seed: int) -> Tuple[float, bool, Dict[str, Any], np.ndarray]:
        """
        Evaluate one GA individual:
          alpha -> w via solve_weights_projected
          train 1 weighted epoch from epoch-e model
          compute delta vs orig e+1
          soft-penalty fit
        Returns:
          (fit, feasible, rec, w_np)
        """
        alpha = _normalize_alpha(alpha.astype(np.float32))

        w_np = solve_weights_projected(
            P=P, target_classes=targets, alpha=alpha,
            w_max=args.w_max, steps=args.opt_steps, seed=int(seed)
        )
        w = torch.from_numpy(w_np).to(device)

        model = ResNet9(num_classes=10).to(device)
        model.load_state_dict(model_e.state_dict())

        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scaler = GradScaler("cuda", enabled=cfg.use_amp)

        train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        scheduler.step()

        after = evaluate_indexed(model, test_loader, cfg)
        after_pc = np.array(after["per_class_acc"], dtype=np.float32)

        delta = after_pc - orig_pc

        # soft penalty fitness
        eps = args.eps
        non_t = [k for k in range(K) if k not in set(targets)]
        neg = delta[non_t][delta[non_t] < 0].sum() if len(non_t) else 0.0

        shortfall = np.minimum(delta[t_arr] - eps, 0.0)  # <=0
        penalty = args.lambda_shortfall * float(shortfall.sum())

        fit = float(delta[t_arr].mean() + neg + penalty)
        feasible = bool(np.all(delta[t_arr] > eps))

        worst_non_t = float(delta[non_t].min()) if len(non_t) else 0.0

        rec = {
            "alpha": alpha.tolist(),
            "fitness": fit,
            "feasible": feasible,
            "orig_e1_overall": float(orig["acc"]),
            "orig_e1_per_class": orig_pc.tolist(),
            "new_e1_overall": float(after["acc"]),
            "new_e1_per_class": after_pc.tolist(),
            "delta_vs_orig_per_class": delta.tolist(),
            "delta_targets": delta[targets].tolist(),
            "mean_target": float(delta[targets].mean()),
            "neg_sum": float(neg),
            "worst_non_target": worst_non_t,
            "shortfall_sum": float(shortfall.sum()),
        }
        return fit, feasible, rec, w_np

    # ===== init population of alphas =====
    pop: List[np.ndarray] = []
    if args.init_from_best_json:
        bj = json_load = None
        try:
            with open(args.init_from_best_json, "r") as f:
                bj = __import__("json").load(f)
            if "alpha" in bj:
                a0 = np.array(bj["alpha"], dtype=np.float32)
                if a0.size == K:
                    pop.append(_normalize_alpha(a0))
                    print("Seeded GA with alpha from:", args.init_from_best_json)
        except Exception as e:
            print("WARN: failed to load init_from_best_json:", e)

    while len(pop) < args.pop:
        a = sample_alpha(K=K, target_classes=targets, rng=rng).astype(np.float32)
        pop.append(_normalize_alpha(a))

    best_rec: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []

    # ===== GA loop =====
    for gen in range(1, args.gens + 1):
        scored: List[Tuple[float, bool, Dict[str, Any], np.ndarray]] = []
        fits = np.empty((args.pop,), dtype=np.float32)

        print(f"\n=== GEN {gen:03d}/{args.gens:03d} ===")
        for i, alpha in enumerate(pop):
            seed_i = int(rng.integers(1e9))
            fit, feasible, rec, w_np = evaluate_alpha(alpha, seed=seed_i)
            fits[i] = fit
            scored.append((fit, feasible, rec, w_np))

            print(
                f"[gen {gen} cand {i}] feasible={feasible} fit={fit:.4f} "
                f"delta_targets={np.array(rec['delta_targets'])} "
                f"mean_target={rec['mean_target']:.4f} neg_sum={rec['neg_sum']:.4f} "
                f"worst_non_target={rec['worst_non_target']:.4f} shortfall_sum={rec['shortfall_sum']:.4f}"
            )

        # sort descending by fitness
        order = np.argsort(-fits)
        scored_sorted = [scored[int(j)] for j in order]

        gen_best_fit, gen_best_feas, gen_best_rec, gen_best_w = scored_sorted[0]

        # update global best
        if best_rec is None or gen_best_fit > float(best_rec["fitness"]):
            best_rec = dict(gen_best_rec)
            best_rec["gen"] = gen
            save_json(os.path.join(args.out_dir, "best.json"), best_rec)
            np.save(os.path.join(args.out_dir, "best_weights.npy"), gen_best_w)
            print("Updated GLOBAL BEST ->", os.path.join(args.out_dir, "best.json"))

        # save per-gen best
        save_json(os.path.join(args.out_dir, f"gen_{gen:03d}_best.json"), gen_best_rec)
        np.save(os.path.join(args.out_dir, f"gen_{gen:03d}_best_weights.npy"), gen_best_w)

        # log generation summary
        feas_count = int(sum(1 for _, feas, _, _ in scored if feas))
        hist_row = {
            "gen": gen,
            "best_fit": float(gen_best_fit),
            "best_feasible": bool(gen_best_feas),
            "feasible_count": feas_count,
            "mean_fit": float(np.mean(fits)),
            "std_fit": float(np.std(fits)),
        }
        history.append(hist_row)
        save_json(os.path.join(args.out_dir, "history.json"), {"history": history})

        # ===== create next generation =====
        elites = [pop[int(order[j])].copy() for j in range(min(args.elite, args.pop))]
        new_pop: List[np.ndarray] = elites.copy()

        while len(new_pop) < args.pop:
            p1 = pop[_tournament_select(rng, fits, k=args.tourn_k)]
            p2 = pop[_tournament_select(rng, fits, k=args.tourn_k)]

            child = _crossover(rng, p1, p2, cx_beta=args.cx_beta)
            child = _mutate(rng, child, mut_rate=args.mut_rate, mut_sigma=args.mut_sigma)
            new_pop.append(child)

        pop = new_pop

    out = {
        "ceiling_evr1": float(evr1),
        "targets": targets,
        "best": best_rec,
        "ga_args": vars(args),
    }
    save_json(os.path.join(args.out_dir, "summary.json"), out)
    print("\nSaved:", args.out_dir)


if __name__ == "__main__":
    main()
