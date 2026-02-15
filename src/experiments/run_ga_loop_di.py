# src/experiments/run_ga_loop_di.py
from __future__ import annotations

import argparse
import os
from typing import Dict, Any, List, Tuple

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
from src.pareto.ga_search import sample_alpha, mutate_alpha
from src.pareto.weight_opt import solve_weights_lp_dual


# -------------------------
# GA utilities (paper: selection/crossover/mutation)
# -------------------------
def _tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int = 3) -> int:
    n = int(fitness.shape[0])
    k = max(1, min(int(k), n))
    cand = rng.integers(0, n, size=k)
    best = int(cand[0])
    best_fit = float(fitness[best])
    for j in cand[1:]:
        jj = int(j)
        if float(fitness[jj]) > best_fit:
            best_fit = float(fitness[jj])
            best = jj
    return best


def _crossover(rng: np.random.Generator, a1: np.ndarray, a2: np.ndarray, beta: float = 0.8) -> np.ndarray:
    # convex mix (keeps alpha in [0,1] if parents in [0,1])
    u = float(rng.uniform(1.0 - beta, beta))
    child = u * a1 + (1.0 - u) * a2
    return np.clip(child, 0.0, 1.0).astype(np.float32)


def _fitness_from_delta(
    delta: np.ndarray,
    target_classes: List[int],
) -> float:
    """
    Paper Algorithm 1 line 7 fitness:
      If ANY target class has Δ <= 0 => fitness = -inf
      Else fitness = (1/|C\\Ctarget|) * sum_{k not in target} 1[Δ_k < 0] * Δ_k
    i.e. maximize (close to 0) non-target degradation, under strict target improvement.
    """
    K = int(delta.shape[0])
    targets = set(int(t) for t in target_classes)

    # hard constraint on targets
    for t in targets:
        if delta[t] <= 0.0:
            return -1e18

    # non-target penalty term (negative or 0)
    non = [k for k in range(K) if k not in targets]
    if len(non) == 0:
        return 0.0
    s = 0.0
    for k in non:
        if delta[k] < 0.0:
            s += float(delta[k])
    return s / float(len(non))


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    # expected format from your training scripts: {"model": state_dict, ...}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def _compute_delta_pct(acc_before: np.ndarray, acc_after: np.ndarray) -> np.ndarray:
    """
    Relative change in performance (percentage):
      Δ_k = 100 * (acc_after - acc_before) / max(eps, acc_before)
    """
    eps = 1e-12
    return (100.0 * (acc_after - acc_before) / np.maximum(eps, acc_before)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()

    # required
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt_e", type=str, required=True, help="checkpoint at epoch e (start point for TRAINONEEPOCH)")
    ap.add_argument("--P_train", type=str, required=True, help="P_train.npy (N x K) computed at epoch e")
    ap.add_argument("--targets", type=str, required=True, help='target classes, e.g. "0,2" or "3"')

    # output
    ap.add_argument("--out_dir", type=str, required=True)

    # training (must match your baseline/noaug setting)
    ap.add_argument("--train_aug", type=int, default=0)  # 0=noaug, 1=train_aug
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)
    # ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])


    # GA
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--tournament_k", type=int, default=3)
    ap.add_argument("--cx_beta", type=float, default=0.8)
    ap.add_argument("--mut_sigma", type=float, default=0.05)

    # LP solver (Algorithm 1 line 4)
    ap.add_argument("--w_max", type=float, default=10.0)
    ap.add_argument("--opt_steps", type=int, default=400)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--lp_lr", type=float, default=0.1)

    # weighted train (Algorithm 1 line 5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--use_amp", type=int, default=1)

    args = ap.parse_args()
    # ---- device auto-selection ----
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", args.device, "| cuda_available=", torch.cuda.is_available())

    ensure_dir(args.out_dir)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    target_classes = [int(x) for x in args.targets.split(",") if x.strip() != ""]
    print("[DI] target_classes =", target_classes)


    # ----- load P -----
    P = np.load(args.P_train)  # [N, K]
    if P.ndim != 2:
        raise ValueError(f"P must be [N,K], got shape {P.shape}")
    N, K = P.shape
    print("[DI] P_train shape:", P.shape)

    # performance ceiling check (paper uses PCA EVR1)
    evr1 = explained_var_ratio_first_pc(P)
    print("[Ceiling check] EVR1:", evr1)

    # ----- data -----
    train_tf = cifar10_train_aug() if int(args.train_aug) == 1 else cifar10_noaug()
    test_tf = cifar10_noaug()

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    train_idx = IndexedDataset(train_ds)
    test_idx = IndexedDataset(test_ds)

    train_loader = DataLoader(
        train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_idx,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- baseline acc at epoch e -----
    base_model = ResNet9(num_classes=K).to(args.device)
    _load_ckpt_into_model(base_model, args.ckpt_e, args.device)

    cfg_eval = WeightedTrainConfig(
        epochs=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        device=args.device,
        num_classes=K,
        use_amp=bool(args.use_amp),
    )
    base_metrics = evaluate_indexed(base_model, test_loader, cfg_eval)
    acc_e = np.array(base_metrics["per_class_acc"], dtype=np.float32)
    # print("[DI] epoch-e per-class:", np.round(acc_e * 100.0, 2))
    print("[DI] ckpt_e per-class acc (%):")
    for k, v in enumerate(acc_e):
        print(f"  class {k:2d}: {v*100:.2f}")
    print(f"[DI] ckpt_e mean acc (%): {acc_e.mean()*100:.2f}")


    # ----- GA init population of alpha -----
    population = [sample_alpha(K=K, target_classes=target_classes, rng=rng) for _ in range(int(args.pop))]

    best: Dict[str, Any] = {
        "fitness": -1e18,
        "alpha": None,
        "w_path": None,
        "delta": None,
        "gen": -1,
    }

    history: List[Dict[str, Any]] = []
    weights_dir = os.path.join(args.out_dir, "weights")
    ensure_dir(weights_dir)

    for g in range(int(args.gens)):
        fits = np.zeros(len(population), dtype=np.float64)
        gen_records: List[Dict[str, Any]] = []

        print(f"\n=== [DI] GEN {g+1:03d}/{int(args.gens):03d} ===")

        for i, alpha in enumerate(population):
            alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
            if alpha.shape[0] != K:
                raise ValueError("alpha shape mismatch")

            # ---- Algorithm 1 line 4: solve LP for w ----
            w = solve_weights_lp_dual(
                P=P,
                target_classes=target_classes,
                alpha=alpha,
                w_max=float(args.w_max),
                steps=int(args.opt_steps),
                lr=float(args.lp_lr),
                seed=int(args.seed + 1000 * g + i),
                tol=float(args.eps),
                normalize_mean_to_1=True,
            ).astype(np.float32)  # [N]

            # save weights for reproducibility
            w_path = os.path.join(weights_dir, f"gen{g:03d}_cand{i:03d}.npy")
            np.save(w_path, w)

            # ---- Algorithm 1 line 5: TRAINONEEPOCH(theta_e, T, w) ----
            model = ResNet9(num_classes=K).to(args.device)
            _load_ckpt_into_model(model, args.ckpt_e, args.device)

            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scaler = GradScaler(enabled=bool(args.use_amp))

            train_one_epoch_weighted(
                model=model,
                loader=train_loader,
                optimizer=opt,
                cfg=cfg_eval,
                scaler=scaler,
                sample_weights=torch.from_numpy(w),
            )

            # ---- evaluate epoch e+1 (weighted) ----
            after_metrics = evaluate_indexed(model, test_loader, cfg_eval)
            acc_e1 = np.array(after_metrics["per_class_acc"], dtype=np.float32)

            # ---- Algorithm 1 line 6: delta (DI uses e -> e+1) ----
            delta = _compute_delta_pct(acc_before=acc_e, acc_after=acc_e1)  # [%]

            fit = _fitness_from_delta(delta=delta, target_classes=target_classes)
            fits[i] = fit

            rec = {
                "gen": g,
                "cand": i,
                "fitness": float(fit),
                "alpha": alpha.tolist(),
                "delta_pct": delta.tolist(),
                "acc_e": acc_e.tolist(),
                "acc_e1": acc_e1.tolist(),
                "w_path": os.path.relpath(w_path, args.out_dir),
            }
            gen_records.append(rec)

            feasible = (fit > -1e17)
            print(
                f"[gen {g+1} cand {i}] feasible={feasible} fit={fit:.6f} "
                f"delta_targets={np.round(delta[target_classes], 4)} "
                f"mean_target={float(delta[target_classes].mean()):.4f} "
                f"worst_non_target={float(np.min(np.delete(delta, target_classes))):.4f}"
            )

            if fit > best["fitness"]:
                best.update(
                    fitness=float(fit),
                    alpha=alpha.tolist(),
                    w_path=os.path.relpath(w_path, args.out_dir),
                    delta=delta.tolist(),
                    gen=g,
                )
                # write best snapshot
                np.save(os.path.join(args.out_dir, "alpha_best.npy"), alpha)
                np.save(os.path.join(args.out_dir, "w_best.npy"), w)
                save_json(os.path.join(args.out_dir, "best.json"), best)

        # save gen logs
        history.append(
            {
                "gen": g,
                "best_fit_in_gen": float(np.max(fits)),
                "mean_fit_in_gen": float(np.mean(fits)),
                "num_feasible": int(np.sum(fits > -1e17)),
                "records": gen_records,
            }
        )
        save_json(os.path.join(args.out_dir, "ga_history.json"), {"history": history, "best": best, "evr1": float(evr1)})

        # ----- GA evolve: elitism + tournament selection + crossover + mutation -----
        elite_n = max(0, min(int(args.elite), len(population)))
        elite_idx = np.argsort(-fits)[:elite_n].tolist()
        new_pop = [population[j] for j in elite_idx]

        while len(new_pop) < int(args.pop):
            p1 = population[_tournament_select(rng, fits, k=int(args.tournament_k))]
            p2 = population[_tournament_select(rng, fits, k=int(args.tournament_k))]
            child = _crossover(rng, np.asarray(p1), np.asarray(p2), beta=float(args.cx_beta))
            child = mutate_alpha(child, sigma=float(args.mut_sigma), rng=rng)
            child = np.clip(child, 0.0, 1.0).astype(np.float32)
            new_pop.append(child)

        population = new_pop[: int(args.pop)]

    print("\n[DI] DONE. Best fitness:", best["fitness"])
    print("[DI] Best alpha saved to:", os.path.join(args.out_dir, "alpha_best.npy"))
    print("[DI] Best weights saved to:", os.path.join(args.out_dir, "w_best.npy"))


if __name__ == "__main__":
    main()
