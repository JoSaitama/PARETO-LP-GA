# src/pareto/weight_opt.py
from __future__ import annotations
import numpy as np


def solve_weights_lp_dual(
    P: np.ndarray,                   # [N, K]
    target_classes: list[int],
    alpha: np.ndarray,               # [K]
    w_max: float = 5.0,
    steps: int = 800,
    lr: float = 0.1,
    seed: int = 0,
    tol: float = 1e-6,
    normalize_mean_to_1: bool = True,
) -> np.ndarray:
    """
    Solve Algorithm-1 Line-4 LP (box-constrained) via a small-dim dual method.

    Primal (paper):
        max_w  sum_{k in target} sum_i w_i P_{i,k}
        s.t.   sum_i w_i P_{i,k} >= alpha_k * sum_i P_{i,k}  for all k in [K]
              0 <= w_i <= w_max

    Notes:
      - No sum(w)=N constraint (paper doesn't include it).
      - K is small (CIFAR10 => 10), so dual variables y in R^K are cheap.
      - For fixed y>=0, primal optimum is:
            w_i = w_max if c_i + (P y)_i > 0 else 0
        where c_i = sum_{k in target} P_{i,k} and (P y)_i = sum_k P_{i,k} y_k.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(P, dtype=np.float64)
    N, K = P.shape
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha.size != K:
        raise ValueError(f"alpha must have shape ({K},), got {alpha.shape}")

    # c_i = sum over target classes
    c = P[:, target_classes].sum(axis=1)  # [N]

    # b_k = alpha_k * sum_i P_{i,k}
    S = P.sum(axis=0)                     # [K]
    b = alpha * S                         # [K]

    # dual vars y >= 0
    y = np.zeros(K, dtype=np.float64)
    # tiny noise to break ties in early iterations (optional)
    y += 1e-6 * rng.standard_normal(K)

    best_w = np.zeros(N, dtype=np.float64)
    best_violation = float("inf")

    # step size schedule (helps stability)
    def step_size(t: int) -> float:
        return lr / np.sqrt(t + 1.0)

    for t in range(int(steps)):
        # scores s_i = c_i + (P y)_i
        s = c + (P @ y)  # [N]

        # primal closed-form under box constraints
        w = np.where(s > 0.0, float(w_max), 0.0).astype(np.float64)

        # check constraint satisfaction
        Aw = P.T @ w  # [K]
        viol = b - Aw
        viol_pos = np.maximum(0.0, viol)
        violation = float(np.max(viol_pos))  # max positive violation

        if violation < best_violation:
            best_violation = violation
            best_w = w.copy()

        if violation <= tol:
            best_w = w
            break

        # subgradient for dual (minimization): g = (A w - b)
        # update y <- [y - eta * (Aw - b)]_+
        eta = step_size(t)
        y = y - eta * (Aw - b)
        y = np.maximum(y, 0.0)

    w_out = best_w.astype(np.float32)

    # Optional: normalize mean weight to ~1 (often stabilizes SGD magnitude)
    if normalize_mean_to_1:
        m = float(w_out.mean())
        if m > 1e-12:
            w_out = w_out / m
            w_out = np.clip(w_out, 0.0, float(w_max)).astype(np.float32)

    return w_out


# Backward-compatible name (used by your run_ga_loop_cc.py)
def solve_weights_projected(
    P: np.ndarray,
    target_classes: list[int],
    alpha: np.ndarray,
    w_max: float = 5.0,
    steps: int = 800,
    lr: float = 0.1,
    penalty: float = 10.0,  # unused (kept for signature compatibility)
    seed: int = 0,
) -> np.ndarray:
    return solve_weights_lp_dual(
        P=P,
        target_classes=target_classes,
        alpha=alpha,
        w_max=w_max,
        steps=steps,
        lr=lr,
        seed=seed,
        tol=1e-6,
        normalize_mean_to_1=False,
    )
