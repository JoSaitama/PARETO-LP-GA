# src/experiments/run_ga_loop_cc.py
from __future__ import annotations

import argparse
import os
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
from src.pareto.ga_search import sample_alpha, mutate_alpha
from src.pareto.weight_opt import solve_weights_lp_dual


def _make_deterministic() -> None:
    # stable enough for GA ranking; avoids cuBLAS determinism crash on some Colab
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    """
    Your ckpt format:
      dict_keys(['epoch','model_state','optimizer_state','scheduler_state','scaler_state','cfg','args'])
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(
            f"Checkpoint format unexpected. Need dict with key 'model_state'. "
            f"Got type={type(ckpt)} keys={list(ckpt.keys()) if isinstance(ckpt, dict) else None}"
        )
    state = ckpt["model_state"]

    # Handle DataParallel prefix "module." if present
    if len(state) > 0:
        k0 = next(iter(state.keys()))
        if k0.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def _print_per_class(acc: np.ndarray, tag: str) -> None:
    print(f"[{tag}] per-class acc (%):")
    for k, v in enumerate(acc):
        print(f"  class {k:2d}: {float(v)*100:.2f}")
    print(f"[{tag}] mean acc (%): {float(acc.mean())*100:.2f}")


def _print_deterioration(acc_e: np.ndarray, acc_e1: np.ndarray, topk: int = 5) -> List[int]:
    """
    Deterioration = acc_e1 - acc_e (most negative first).
    Return top-k most deteriorated classes.
    """
    delta = acc_e1 - acc_e
    order = np.argsort(delta)  # ascending: most negative first

    print("[AutoTargets] deterioration acc(e+1)-acc(e) (%): (most negative first)")
    for rank, cls in enumerate(order[: min(topk, len(order))]):
        print(f"  rank {rank+1:2d}: class {int(cls):2d}  delta={float(delta[int(cls)])*100:.2f}")

    return [int(i) for i in order[: min(topk, len(order))]]


def pick_cc_targets_from_ckpts(
    model_e: torch.nn.Module,
    model_e1: torch.nn.Module,
    test_loader: DataLoader,
    cfg: WeightedTrainConfig,
    top_n: int = 2,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick targets as most deteriorated classes from epoch e -> original epoch e+1.
    """
    res_e = evaluate_indexed(model_e, test_loader, cfg)
    res_e1 = evaluate_indexed(model_e1, test_loader, cfg)

    acc_e = np.array(res_e["per_class_acc"], dtype=np.float32)
    acc_e1 = np.array(res_e1["per_class_acc"], dtype=np.float32)
    delta = acc_e1 - acc_e

    order = np.argsort(delta)  # most negative first
    top_n = int(max(1, min(top_n, acc_e.size)))
    targets = [int(i) for i in order[:top_n]]
    return targets, acc_e, acc_e1, delta


def _compute_delta_pct(acc_before: np.ndarray, acc_after: np.ndarray) -> np.ndarray:
    """
    Relative change (%):
      Δ_k = 100 * (after - before) / max(eps, before)
    """
    eps = 1e-12
    return (100.0 * (acc_after - acc_before) / np.maximum(eps, acc_before)).astype(np.float32)


def _fitness_from_delta(delta_pct: np.ndarray, target_classes: List[int]) -> float:
    """
    Paper Algorithm 1 (fitness):
      If ANY target class has Δ <= 0 => infeasible => very negative
      Else fitness = (1/|non-target|) * sum_{k not in target, Δ_k < 0} Δ_k
    """
    K = int(delta_pct.shape[0])
    targets = set(int(t) for t in target_classes)

    for t in targets:
        if float(delta_pct[t]) <= 0.0:
            return -1e18

    non = [k for k in range(K) if k not in targets]
    if len(non) == 0:
        return 0.0

    s = 0.0
    for k in non:
        if float(delta_pct[k]) < 0.0:
            s += float(delta_pct[k])
    return s / float(len(non))


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
    u = float(rng.uniform(1.0 - beta, beta))
    child = u * a1 + (1.0 - u) * a2
    return np.clip(child, 0.0, 1.0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, required=True)

    # CC needs BOTH ckpt_e and original ckpt_{e+1}
    p.add_argument("--ckpt_e", type=str, required=True)
    p.add_argument("--ckpt_orig_e1", type=str, required=True)

    p.add_argument("--P_train", type=str, required=True)

    # allow empty if using auto_targets
    p.add_argument("--targets", type=str, default="")
    p.add_argument("--auto_targets", type=int, default=0)
    p.add_argument("--auto_topn", type=int, default=2)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--train_aug", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # PARETO stage optimizer (paper table often uses small lr/wd; you can override)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--use_amp", type=int, default=1)

    # LP solver (paper line 4)
    p.add_argument("--w_max", type=float, default=5.0)
    p.add_argument("--opt_steps", type=int, default=800)
    p.add_argument("--lp_lr", type=float, default=0.1)
    p.add_argument("--lp_tol", type=float, default=1e-6)

    # GA
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=12)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--tournament_k", type=int, default=3)
    p.add_argument("--cx_beta", type=float, default=0.7)
    p.add_argument("--mut_sigma", type=float, default=0.10)

    args = p.parse_args()

    set_seed(args.seed)
    _make_deterministic()
    rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", args.device, "| cuda_available=", torch.cuda.is_available())

    ensure_dir(args.out_dir)

    # ----- load P -----
    P = np.load(args.P_train)
    if P.ndim != 2:
        raise ValueError(f"P_train must be [N,K], got {P.shape}")
    N, K = P.shape
    print("[CC] P_train shape:", P.shape)

    evr1 = explained_var_ratio_first_pc(P)
    print("[Ceiling check] EVR1:", float(evr1))

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

    cfg = WeightedTrainConfig(
        epochs=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        device=args.device,
        num_classes=K,
        use_amp=bool(args.use_amp) and (args.device == "cuda"),
    )

    # ----- load ckpt_e and ckpt_orig_e1, compute acc tables -----
    model_e = ResNet9(num_classes=K).to(args.device)
    _load_ckpt_into_model(model_e, args.ckpt_e, args.device)

    model_orig_e1 = ResNet9(num_classes=K).to(args.device)
    _load_ckpt_into_model(model_orig_e1, args.ckpt_orig_e1, args.device)

    # per-class acc for convenience
    res_e = evaluate_indexed(model_e, test_loader, cfg)
    acc_e = np.array(res_e["per_class_acc"], dtype=np.float32)
    _print_per_class(acc_e, "CC ckpt_e")

    res_orig_e1 = evaluate_indexed(model_orig_e1, test_loader, cfg)
    acc_orig_e1 = np.array(res_orig_e1["per_class_acc"], dtype=np.float32)
    _print_per_class(acc_orig_e1, "CC ckpt_orig_e1")

    # suggest/auto pick targets
    if int(args.auto_targets) == 1:
        targets, _, _, _ = pick_cc_targets_from_ckpts(
            model_e=model_e,
            model_e1=model_orig_e1,
            test_loader=test_loader,
            cfg=cfg,
            top_n=int(args.auto_topn),
        )
        print("[CC][AutoTargets] picked targets:", targets)
        _print_deterioration(acc_e, acc_orig_e1, topk=K)
        target_classes = targets
    else:
        if args.targets.strip() == "":
            raise ValueError("You must provide --targets or set --auto_targets 1")
        target_classes = [int(x) for x in args.targets.split(",") if x.strip() != ""]
        print("[CC] target_classes =", target_classes)

    # ----- GA init -----
    population = [sample_alpha(K=K, target_classes=target_classes, rng=rng) for _ in range(int(args.pop))]

    best: Dict[str, Any] = {
        "fitness": -1e18,
        "alpha": None,
        "w_path": None,
        "delta_pct": None,
        "gen": -1,
    }

    history: List[Dict[str, Any]] = []
    weights_dir = os.path.join(args.out_dir, "weights")
    ensure_dir(weights_dir)

    for g in range(int(args.gens)):
        fits = np.zeros(len(population), dtype=np.float64)
        gen_records: List[Dict[str, Any]] = []

        print(f"\n=== [CC] GEN {g+1:03d}/{int(args.gens):03d} ===")

        for i, alpha in enumerate(population):
            alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
            if alpha.shape[0] != K:
                raise ValueError("alpha shape mismatch")

            # ----- LP solve w (paper line 4) -----
            w = solve_weights_lp_dual(
                P=P,
                target_classes=target_classes,
                alpha=alpha,
                w_max=float(args.w_max),
                steps=int(args.opt_steps),
                lr=float(args.lp_lr),
                seed=int(args.seed + 1000 * g + i),
                tol=float(args.lp_tol),
                normalize_mean_to_1=True,
            ).astype(np.float32)

            w_path = os.path.join(weights_dir, f"gen{g:03d}_cand{i:03d}.npy")
            np.save(w_path, w)

            # ----- train one epoch from theta_e with weights -----
            model = ResNet9(num_classes=K).to(args.device)
            _load_ckpt_into_model(model, args.ckpt_e, args.device)

            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scaler = GradScaler(enabled=bool(args.use_amp) and (args.device == "cuda"))

            train_one_epoch_weighted(
                model=model,
                loader=train_loader,
                optimizer=opt,
                cfg=cfg,
                scaler=scaler,
                sample_weights=torch.from_numpy(w),
            )

            # ----- eval new epoch e+1 -----
            res_new = evaluate_indexed(model, test_loader, cfg)
            acc_new_e1 = np.array(res_new["per_class_acc"], dtype=np.float32)

            # ----- CC delta: orig e+1 vs new weighted e+1 -----
            delta_pct = _compute_delta_pct(acc_before=acc_orig_e1, acc_after=acc_new_e1)

            fit = _fitness_from_delta(delta_pct=delta_pct, target_classes=target_classes)
            fits[i] = fit

            feasible = (fit > -1e17)
            print(
                f"[gen {g+1} cand {i}] feasible={feasible} fit={fit:.6f} "
                f"delta_targets={np.round(delta_pct[target_classes], 4)} "
                f"mean_target={float(delta_pct[target_classes].mean()):.4f} "
                f"worst_non_target={float(np.min(np.delete(delta_pct, target_classes))):.4f}"
            )

            rec = {
                "gen": g,
                "cand": i,
                "fitness": float(fit),
                "alpha": alpha.tolist(),
                "delta_pct": delta_pct.tolist(),
                "w_path": os.path.relpath(w_path, args.out_dir),
            }
            gen_records.append(rec)

            if fit > best["fitness"]:
                best.update(
                    fitness=float(fit),
                    alpha=alpha.tolist(),
                    w_path=os.path.relpath(w_path, args.out_dir),
                    delta_pct=delta_pct.tolist(),
                    gen=g,
                )
                np.save(os.path.join(args.out_dir, "alpha_best.npy"), alpha)
                np.save(os.path.join(args.out_dir, "w_best.npy"), w)
                save_json(os.path.join(args.out_dir, "best.json"), best)

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

        # ----- evolve -----
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

    print("\n[CC] DONE. Best fitness:", best["fitness"])
    print("[CC] Best alpha saved to:", os.path.join(args.out_dir, "alpha_best.npy"))
    print("[CC] Best weights saved to:", os.path.join(args.out_dir, "w_best.npy"))


if __name__ == "__main__":
    main()
