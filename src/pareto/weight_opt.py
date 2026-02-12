# src/pareto/weight_opt.py
from __future__ import annotations
import numpy as np

def solve_weights_projected(
    P: np.ndarray,                   # [N, K]
    target_classes: list[int],
    alpha: np.ndarray,               # [K], in [0, 1]
    w_max: float = 5.0,
    steps: int = 400,
    lr: float = 0.05,
    penalty: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """
    A practical alternative to large-scale LP:
    maximize c^T w  subject to  A w >= b, sum w = N, 0<=w<=w_max
    where A_k,i = P_i,k ; b_k = alpha_k * sum_i P_i,k ; c_i = sum_{k in target} P_i,k
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(P, dtype=np.float64)
    N, K = P.shape

    # objective vector c
    c = P[:, target_classes].sum(axis=1)  # [N]

    # constraints
    S = P.sum(axis=0)                      # [K]
    b = alpha * S                          # [K]

    w = np.ones(N, dtype=np.float64)

    # small noise helps escape flat region
    w += 0.01 * rng.standard_normal(N)
    w = np.clip(w, 0.0, w_max)
    w *= (N / max(1e-12, w.sum()))

    for _ in range(steps):
        Aw = P.T @ w                       # [K]
        viol = np.maximum(0.0, b - Aw)     # [K] (positive means violation)

        # gradient ascent with penalty for violations
        grad = c.copy()                    # [N]
        if viol.max() > 0:
            grad += penalty * (P @ viol)   # [N]

        w = w + lr * grad
        w = np.clip(w, 0.0, w_max)

        # project sum(w)=N
        s = w.sum()
        if s <= 1e-12:
            w[:] = 1.0
            s = w.sum()
        w *= (N / s)

    return w.astype(np.float32)

