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
from src.pareto.weight_opt import solve_weights_projected


def _project_alpha(alpha: np.ndarray, mode: str = "clip") -> np.ndarray:
    """
    Project alpha according to selected mode.

    clip     : clip each alpha_k to [0,1]  (paper-consistent per-class thresholds)
    none     : leave alpha unchanged (cast to float32)
    simplex  : old repo behavior (nonnegative & sum=1) kept for ablation
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
    """Return index of winner among k random indices."""
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
    """
    Blend crossover:
      child = u*a1 + (1-u)*a2
      u ~ Uniform(1-beta, beta)

    Note: This makes most sense when beta in [0.5, 1.0].
    If user passes beta<0.5, we clamp to 0.5 to avoid inverted interval.
    """
    beta = float(cx_beta)
    if beta < 0.5:
        beta = 0.5
    if beta > 1.0:
        beta = 1.0
    lo = 1.0 - beta
    hi = beta
    u = float(rng.uniform(lo, hi))
    child = u * a1 + (1.0 - u) * a2
    return _project_alpha(child, alpha_project)


def _mutate(
    rng: np.random.Generator,
    a: np.ndarray,
    mut_rate: float,
    mut_sigma: float,
    alpha_project: str,
) -> np.ndarray:
    """Gaussian mutation on a subset of dims, then project."""
    child = a.astype(np.float32).copy()
    K = int(child.size)
    # number of mutated dims
    m = int(np.ceil(float(mut_rate) * K))
    m = max(1, min(K, m))
    idx = rng.choice(K, size=m, replace=False)
    child[idx] += rng.normal(loc=0.0, scale=float(mut_sigma), size=m).astype(np.float32)
    return _project_alpha(child, alpha_project)


def main():
    p = argparse.ArgumentParser()

    # ===== core args =====
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
    p.add_argument("--init_from_best_json", type=str, default="")  # optional: seed population with alpha from a prior run

    # IMPORTANT: new flag to control alpha projection (paper-consistent default: clip)
    p.add_argument(
        "--alpha_project",
        type=str,
        default="clip",
        choices=["clip", "none", "simplex"],
        help="How to project alpha after mutation/crossover/evaluation. clip is paper-consistent.",
    )

    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    ensure_dir(args.out_dir)

    targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
    if len(targets) == 0:
        raise ValueEr
