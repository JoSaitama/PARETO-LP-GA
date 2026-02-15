# src/pareto/weight_opt_paper_lp.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class LPSolveDiag:
    max_viol: float
    sum_viol: float
    obj: float
    selected_ratio: float
    y_norm: float
    steps_run: int


def solve_weights_lp_dual_paper(
    P: np.ndarray,                 # [N, K]
    target_classes: list[int],     # T
    alpha: np.ndarray,             # [K]
    w_max: float = 10.0,
    steps: int = 2000,
    lr: float = 0.5,
    seed: int = 0,
    tol: float = 1e-6,
    p_scale: str = "maxabs",       # {"none","maxabs"}
    p_flip: bool = False,          # only if you KNOW your P sign is opposite
    return_diag: bool = False,
) -> tuple[np.ndarray, LPSolveDiag] | np.ndarray:
    """
    Implements the LP in PARETO-LP-GA Algorithm 1, Line 4:

    Primal (paper):
        maximize_w   sum_{i} w_i * (sum_{t in T} P_{i,t})
        subject to   sum_{i} w_i P_{i,k} >= alpha_k * sum_{i} P_{i,k}   for all k=1..K
                     0 <= w_i <= w_max

    Dual / solver:
      - Uses projected subgradient on y >= 0.
      - Given y, primal optimum is:
            w_i = w_max if c_i + (P y)_i > 0 else 0
        where c_i = sum_{t in T} P_{i,t}   (NO alpha in objective, matching paper).

    Selection of best iterate (IMPORTANT):
      We pick the iterate with smallest max violation first,
      then (among ties / near-ties) the largest primal objective.
      This avoids "always selecting the same w" across different alphas.
    """
    rng = np.random.default_rng(seed)

    P = np.asarray(P, dtype=np.float64)
    if p_flip:
        P = -P

    # Optional scaling for numerical stability (keeps sign structure)
    if p_scale == "maxabs":
        s = float(np.max(np.abs(P)) + 1e-12)
        P = P / s
    elif p_scale == "none":
        pass
    else:
        raise ValueError("p_scale must be 'maxabs' or 'none'")

    N, K = P.shape
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha.shape[0] != K:
        raise ValueError(f"alpha must have shape ({K},), got {alpha.shape}")

    T = list(target_classes)
    if len(T) == 0:
        w = np.ones(N, dtype=np.float32)
        diag = LPSolveDiag(0.0, 0.0, 0.0, 1.0, 0.0, 0)
        return (w, diag) if return_diag else w

    # ===== Algorithm 1 Line 4 components =====
    # c_i = sum_{t in T} P_{i,t}   (paper objective)
    c = P[:, T].sum(axis=1)  # [N]

    # S_k = sum_i P_{i,k}
    S = P.sum(axis=0)        # [K]

    # b_k = alpha_k * S_k
    b = alpha * S            # [K]

    # ===== Projected subgradient on dual y >= 0 =====
    y = np.zeros(K, dtype=np.float64)

    # tiny init noise to break ties; keep y >= 0
    y += 1e-8 * rng.standard_normal(K)
    y = np.maximum(y, 0.0)

    # Best iterate tracking with lexicographic criterion:
    # (max_viol, -obj, sum_viol) minimal.
    best_key = (float("inf"), float("inf"), float("inf"))
    best_w = np.zeros(N, dtype=np.float64)
    best_diag = None

    def eval_w(w: np.ndarray) -> tuple[tuple[float, float, float], LPSolveDiag]:
        # Aw_k = sum_i w_i P_{i,k}
        Aw = P.T @ w  # [K]
        viol = b - Aw
        viol_pos = np.maximum(0.0, viol)

        max_viol = float(np.max(viol_pos))
        sum_viol = float(np.sum(viol_pos))

        # primal objective: sum_i w_i * c_i
        obj = float(np.dot(w, c))

        key = (max_viol, -obj, sum_viol)
        diag = LPSolveDiag(
            max_viol=max_viol,
            sum_viol=sum_viol,
            obj=obj,
            selected_ratio=float(np.mean(w > 0)),
            y_norm=float(np.linalg.norm(y)),
            steps_run=0,
        )
        return key, diag

    # step schedule: stable + not too tiny
    def step_size(t: int) -> float:
        # this works better than 1/sqrt(t) when you need movement
        return lr / (1.0 + 0.01 * t)

    for t in range(int(steps)):
        # s_i = c_i + (P y)_i
        s = c + (P @ y)  # [N]

        # exact primal optimizer under 0<=w<=w_max
        # (tie-breaking jitter avoids deterministic "same w" when s is near 0)
        jitter = 1e-12 * rng.standard_normal(N)
        w = np.where(s + jitter > 0.0, float(w_max), 0.0).astype(np.float64)

        # Evaluate and keep best
        key, diag = eval_w(w)
        diag.steps_run = t + 1
        if key < best_key:
            best_key = key
            best_w = w.copy()
            best_diag = diag

        # If sufficiently feasible, you can early stop
        if diag.max_viol <= tol:
            break

        # Dual subgradient: g = (b - Aw)
        Aw = P.T @ w
        g = (b - Aw)

        eta = step_size(t)
        y = y + eta * g
        y = np.maximum(y, 0.0)

    w_out = best_w.astype(np.float32)
    if best_diag is None:
        best_diag = LPSolveDiag(float("inf"), float("inf"), 0.0, 0.0, float(np.linalg.norm(y)), int(steps))

    return (w_out, best_diag) if return_diag else w_out
