# src/experiments/run_ga_loop_cc.py
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
from src.pareto.weight_opt import solve_weights_projected,solve_weights_soft


def _project_alpha(alpha: np.ndarray, mode: str = "clip") -> np.ndarray:
    """
    clip     : clip each alpha_k to [0,1]  (paper-consistent threshold range)
    none     : leave alpha unchanged
    simplex  : nonnegative & sum=1 (old repo behavior, for ablation only)
    """
    a = alpha.astype(np.float32)

    if mode == "clip":
        return np.clip(a, 0.0, 2.0)

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
    beta = min(max(beta, 0.5), 1.0)  # clamp to [0.5, 1.0]
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


def pick_cc_targets_from_ckpts(
    model_e: torch.nn.Module,
    model_e1: torch.nn.Module,
    test_loader: DataLoader,
    cfg: WeightedTrainConfig,
    top_n: int = 2,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    res_e = evaluate_indexed(model_e, test_loader, cfg)
    res_e1 = evaluate_indexed(model_e1, test_loader, cfg)

    acc_e = np.array(res_e["per_class_acc"], dtype=np.float32)
    acc_e1 = np.array(res_e1["per_class_acc"], dtype=np.float32)
    delta = acc_e1 - acc_e

    order = np.argsort(delta)  # most negative first
    top_n = int(max(1, min(top_n, acc_e.size)))
    targets = [int(i) for i in order[:top_n]]
    return targets, acc_e, acc_e1, delta


def _make_deterministic():
    # stable enough for GA ranking; avoids cuBLAS determinism crash on Colab
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DO NOT call torch.use_deterministic_algorithms(True) on Colab unless
    # you set CUBLAS_WORKSPACE_CONFIG before importing torch.
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt_e", type=str, required=True)
    p.add_argument("--ckpt_orig_e1", type=str, required=True)
    p.add_argument("--P_train", type=str, required=True)

    # allow empty when using --auto_targets
    p.add_argument("--targets", type=str, default="")
    p.add_argument("--auto_targets", type=int, default=0)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    # Paper is strict: target must improve (>0). We'll keep eps as margin.
    p.add_argument("--eps", type=float, default=0.0)

    p.add_argument("--w_max", type=float, default=5.0)
    p.add_argument("--opt_steps", type=int, default=800)

    # GA
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=12)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--tourn_k", type=int, default=3)
    p.add_argument("--cx_beta", type=float, default=0.7)
    p.add_argument("--mut_rate", type=float, default=0.3)
    p.add_argument("--mut_sigma", type=float, default=0.10)
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

    # ----------------------------
    # Fix influence sign to match paper's LP direction
    # Paper assumes larger P is "better" for targets in the LP objective/constraints.
    # If sums over classes are mostly negative, constraints become trivial and w becomes alpha-invariant.
    # ----------------------------
    S = P.sum(axis=0)  # [K]
    print("[P_check] sum(P) per class:", S)
    print("[P_check] mean sum(P):", float(S.mean()))
    
    # Heuristic: if average class sum is negative, flip sign
    if float(S.mean()) < 0.0:
        print("[P_check] Flipping P <- -P to match LP direction.")
        P = -P
        S2 = P.sum(axis=0)
        print("[P_check] After flip, mean sum(P):", float(S2.mean()))

    
    evr1 = explained_var_ratio_first_pc(P)
    print("Ceiling check EVR1:", evr1)

    train_tf = cifar10_train_aug() if args.train_aug else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_idx_ds = IndexedDataset(train_ds)
    test_idx_ds = IndexedDataset(test_ds)

    # IMPORTANT: during GA evaluation, fix the train order to reduce noise
    train_loader = DataLoader(
        train_idx_ds,
        batch_size=args.batch_size,
        shuffle=False,
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

    model_e = ResNet9(num_classes=K).to(device)
    model_orig = ResNet9(num_classes=K).to(device)

    ckpt_e = torch.load(args.ckpt_e, map_location=device)
    ckpt_o = torch.load(args.ckpt_orig_e1, map_location=device)
    model_e.load_state_dict(ckpt_e["model_state"])
    model_orig.load_state_dict(ckpt_o["model_state"])

    # targets
    if int(args.auto_targets) > 0:
        targets, acc_e, acc_e1, delta_e1_minus_e = pick_cc_targets_from_ckpts(
            model_e=model_e,
            model_e1=model_orig,
            test_loader=test_loader,
            cfg=cfg,
            top_n=int(args.auto_targets),
        )
        print("\n[AutoTargets] Picked CC targets as most deteriorated classes (e -> e+1):", targets)
        print("[AutoTargets] epoch-e per-class:", acc_e)
        print("[AutoTargets] epoch-(e+1) per-class:", acc_e1)
        print("[AutoTargets] delta (e+1 - e):", delta_e1_minus_e)
    else:
        targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
        if len(targets) == 0:
            raise ValueError("targets is empty. Use --targets like '2,3', or set --auto_targets > 0.")

    orig = evaluate_indexed(model_orig, test_loader, cfg)
    orig_pc = np.array(orig["per_class_acc"], dtype=np.float32)
    print("Orig e+1 overall:", orig["acc"])
    print("Orig e+1 per-class:", orig_pc)

    t_arr = np.array(targets, dtype=int)

    def evaluate_alpha(alpha: np.ndarray) -> Tuple[float, bool, Dict[str, Any], np.ndarray]:
        alpha = _project_alpha(alpha, args.alpha_project)

        # solve Line-4 LP (now real LP solver behind this call)
        # w_np = solve_weights_projected(
        #     P=P,
        #     target_classes=targets,
        #     alpha=alpha,
        #     w_max=args.w_max,
        #     steps=args.opt_steps,
        #     seed=int(args.seed),  # keep fixed for stable fitness ranking
        # )
        w_np,_ = solve_weights_soft(P, targets, alpha, w_max=float(args.w_max))
        # DEBUG
        if (not np.all(np.isfinite(w_np))) or (w_np.ndim != 1):
            rec = {"alpha": alpha.tolist(), "fitness": -1e6, "feasible": False,
                   "delta_targets": [float("nan") for _ in targets],
                   "mean_target": float("nan"), "neg_sum": float("nan"),
                   "worst_non_target": float("nan"), "shortfall_sum": float("nan")}
            return -1e6, False, rec, np.zeros(P.shape[0], dtype=np.float32)
        print("for debug w_stats:",
            float(w_np.min()), float(w_np.max()), float(w_np.mean()),
            "nz%", float((w_np > 0).mean()),
            "at_wmax%", float((w_np >= float(args.w_max) - 1e-6).mean()))
        # for debug
        print("alpha_in_eval:", np.array(alpha, dtype=np.float32))
        w = torch.from_numpy(w_np).to(device)
        # for debug
        w = torch.ones_like(w)
        
        # # train one weighted epoch from epoch-e
        # model = ResNet9(num_classes=K).to(device)
        # model.load_state_dict(model_e.state_dict())

        # optimizer = optim.SGD(
        #     model.parameters(),
        #     lr=cfg.lr,
        #     momentum=cfg.momentum,
        #     weight_decay=cfg.weight_decay,
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        # scaler = GradScaler("cuda", enabled=cfg.use_amp)

        # train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        # scheduler.step()

        # =========================
        # A) TRAINONEEPOCH: restore full training state from ckpt_e
        # =========================
        model = ResNet9(num_classes=K).to(device)
        model.load_state_dict(ckpt_e["model_state"])   # <- IMPORTANT: load from ckpt dict, not model_e copy
        
        # Rebuild optimizer/scheduler exactly like run_train_snapshots.py
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(ckpt_e.get("cfg", {}).get("lr", cfg.lr)),
            momentum=float(ckpt_e.get("cfg", {}).get("momentum", cfg.momentum)),
            weight_decay=float(ckpt_e.get("cfg", {}).get("weight_decay", cfg.weight_decay)),
        )
        
        # Load optimizer state if present (should be present if ckpt made by run_train_snapshots.py)
        if "optimizer_state" in ckpt_e and ckpt_e["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt_e["optimizer_state"])
        
        # Scheduler must match training script: CosineAnnealingLR(T_max=cfg.epochs)
        tmax = int(ckpt_e.get("cfg", {}).get("epochs", 1))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, tmax))
        
        if "scheduler_state" in ckpt_e and ckpt_e["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt_e["scheduler_state"])
        
        # AMP scaler restore
        scaler = GradScaler("cuda", enabled=cfg.use_amp)
        if cfg.use_amp and ("scaler_state" in ckpt_e) and (ckpt_e["scaler_state"] is not None):
            scaler.load_state_dict(ckpt_e["scaler_state"])
        
        # One weighted epoch, then step scheduler exactly once (like baseline e -> e+1)
        train_one_epoch_weighted(model, train_loader, optimizer, cfg, scaler, w)
        scheduler.step()

    
        after = evaluate_indexed(model, test_loader, cfg)
        after_pc = np.array(after["per_class_acc"], dtype=np.float32)
        delta = after_pc - orig_pc

        # eps = float(args.eps)

        # # ===== paper-consistent feasibility: all targets must strictly improve (>= eps) =====
        # # feasible = bool(np.all(delta[t_arr] >= eps))

        # # ===== paper-consistent fitness (Line 7):
        # # if any target doesn't improve -> -inf (we use a large negative)
        # # else sum of negative deltas on non-target (closer to 0 is better)
        # target_set = set(targets)
        # non_t = [k for k in range(K) if k not in target_set]

        # neg_sum = float(delta[non_t][delta[non_t] < 0].sum()) if len(non_t) else 0.0

        # # if not feasible:
        # #     fit = -1e9
        # # else:
        # #     fit = neg_sum  # maximize (best is 0, worst more negative)

        # # shortfall <= 0 means not meeting eps on targets
        # shortfall = np.minimum(delta[t_arr] - eps, 0.0).astype(np.float32)
        
        eps = float(args.eps)
        
        # =========================
        # B) Fitness exactly as paper Algorithm 1 Line 7
        #   - feasible: all target deltas must be > 0 (or >= eps if you set eps)
        #   - if infeasible -> -inf (use large negative)
        #   - else fitness = mean of negative deltas over non-target classes (best is 0)
        # =========================
        t_arr = np.array(targets, dtype=np.int64)
        
        # paper uses "positive improvement"; with eps=0.0 this is strict >0.
        # If you want strict, use: delta[t_arr] > eps
        feasible = bool(np.all(delta[t_arr] > eps))
        
        target_set = set(targets)
        non_t = [k for k in range(K) if k not in target_set]
        
        neg = delta[non_t][delta[non_t] < 0] if len(non_t) else np.array([], dtype=np.float32)
        neg_sum = float(neg.sum()) if neg.size else 0.0
        
        if not feasible:
            fit = -1e9
        else:
            fit = float(neg.mean()) if neg.size else 0.0   # maximize (closest to 0 is best)



        worst_non_t = float(delta[non_t].min()) if len(non_t) else 0.0

        rec: Dict[str, Any] = {
            "alpha": alpha.tolist(),
            "fitness": float(fit),
            "feasible": bool(feasible),
            "orig_e1_overall": float(orig["acc"]),
            "orig_e1_per_class": orig_pc.tolist(),
            "new_e1_overall": float(after["acc"]),
            "new_e1_per_class": after_pc.tolist(),
            "delta_vs_orig_per_class": delta.tolist(),
            "delta_targets": delta[targets].tolist(),
            "mean_target": float(delta[targets].mean()),
            "neg_sum": float(neg_sum),
            "worst_non_target": float(worst_non_t),
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
        a = sample_alpha(K=K, target_classes=targets, rng=rng).astype(np.float32)
        pop.append(_project_alpha(a, args.alpha_project))

    best_rec: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []

    for gen in range(1, int(args.gens) + 1):
        scored: List[Tuple[float, bool, Dict[str, Any], np.ndarray]] = []
        fits = np.empty((int(args.pop),), dtype=np.float32)

        print(f"\n=== GEN {gen:03d}/{int(args.gens):03d} ===")
        for i, alpha in enumerate(pop):
            fit, feasible, rec, w_np = evaluate_alpha(alpha)
            fits[i] = fit
            scored.append((fit, feasible, rec, w_np))

            print(
                f"[gen {gen} cand {i}] feasible={feasible} fit={fit:.4f} "
                f"delta_targets={np.array(rec['delta_targets'])} "
                f"mean_target={rec['mean_target']:.4f} neg_sum={rec['neg_sum']:.4f} "
                f"worst_non_target={rec['worst_non_target']:.4f}"
            )

            if i < 5:
                print("alpha_stats", float(alpha.min()), float(alpha.max()), float(alpha.mean()))

        order = np.argsort(-fits)  # descending
        scored_sorted = [scored[int(j)] for j in order]
        gen_best_fit, gen_best_feas, gen_best_rec, gen_best_w = scored_sorted[0]

        if best_rec is None or float(gen_best_fit) > float(best_rec["fitness"]):
            best_rec = dict(gen_best_rec)
            best_rec["gen"] = gen
            save_json(os.path.join(args.out_dir, "best.json"), best_rec)
            np.save(os.path.join(args.out_dir, "best_weights.npy"), gen_best_w)
            print("Updated GLOBAL BEST ->", os.path.join(args.out_dir, "best.json"))

        save_json(os.path.join(args.out_dir, f"gen_{gen:03d}_best.json"), gen_best_rec)
        np.save(os.path.join(args.out_dir, f"gen_{gen:03d}_best_weights.npy"), gen_best_w)

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

        elites = [pop[int(order[j])].copy() for j in range(min(int(args.elite), int(args.pop)))]
        new_pop: List[np.ndarray] = elites.copy()

        while len(new_pop) < int(args.pop):
            p1 = pop[_tournament_select(rng, fits, k=int(args.tourn_k))]
            p2 = pop[_tournament_select(rng, fits, k=int(args.tourn_k))]
            child = _crossover(rng, p1, p2, cx_beta=args.cx_beta, alpha_project=args.alpha_project)
            child = _mutate(rng, child, mut_rate=args.mut_rate, mut_sigma=args.mut_sigma, alpha_project=args.alpha_project)
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
