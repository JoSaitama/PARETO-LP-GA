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
    
    # P = -P  # DO NOT flip here for EKFAC-based P_train (already beneficial-positive)
    N, K = P.shape
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha.size != K:
        raise ValueError(f"alpha must have shape ({K},), got {alpha.shape}")

    # c_i = sum over target classes
    c = P[:, target_classes].sum(axis=1)  # [N]

    # b_k = alpha_k * sum_i P_{i,k}
    S = P.sum(axis=0)                     # [K]
    b = alpha * S                         # [K]

    print("[LP_debug] alpha min/max:", float(alpha.min()), float(alpha.max()))
    print("[LP_debug] b min/max/mean:", float(b.min()), float(b.max()), float(b.mean()))
    print("[LP_debug] S per class:", S)

    
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
        s = c + (P @ y)  # [N]  # correct sign: c + A^T y
        # s = c - (P @ y)  # old (wrong sign)

        # primal closed-form under box constraints
        # primal closed-form under box constraints (paper): 0 <= w_i <= w_max
        w = np.where(s > 0.0, float(w_max), 0.0).astype(np.float64)
        # w = np.where(s < 0.0, float(w_max), 1.0).astype(np.float64)

        # check constraint satisfaction
        Aw = P.T @ w  # [K]
        viol = b - Aw
        viol_pos = np.maximum(0.0, viol)
        violation = float(np.max(viol_pos))  # max positive violation
        S  = P.sum(axis=0)
        ratio = Aw / (S + 1e-12)
        # print("[DBG] achieved_ratio min/mean/max:",
        #       float(ratio.min()), float(ratio.mean()), float(ratio.max()))
        
        if violation < best_violation:
            best_violation = violation
            best_w = w.copy()

        if (t > 200) and (violation <= tol):
            best_w = w
            break

        # subgradient for dual (minimization): g = (A w - b)
        # update y <- [y - eta * (Aw - b)]_+
        eta = step_size(t)
        y = y - eta * (Aw - b)
        y = np.maximum(y, 0.0)
    
    print("[LP_debug] best_violation:", best_violation)
    print("[LP_debug] y_norm:", float(np.linalg.norm(y)))
    print("[LP_debug] selected_ratio:", float(np.mean(best_w > 0)))
    

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
        normalize_mean_to_1=True,
    )

def solve_weights_hard_topk(
    P: np.ndarray,
    target_classes: list,
    alpha: np.ndarray,
    w_max: float,
    keep_ratio: float = 0.2,
):
    """
    Paper-like hard reweighting:
      - score_i = sum_{t in targets} alpha_t * P_{i,t}
      - keep top-k (k = ceil(keep_ratio * N)) samples
      - w_i = w_max for kept, else 1.0
    """
    P = np.asarray(P, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)

    N, K = P.shape
    targets = list(target_classes)
    if len(targets) == 0:
        return np.ones(N, dtype=np.float32)

    # alpha 可能是全K维，也可能你只传targets维；这里两种都兼容
    if alpha.shape[0] == K:
        a_t = alpha[targets]
    else:
        # assume alpha aligned with targets
        if alpha.shape[0] != len(targets):
            raise ValueError(f"alpha dim {alpha.shape[0]} not match K={K} or len(targets)={len(targets)}")
        a_t = alpha

    # score per sample
    score = P[:, targets] @ a_t  # [N]

    k = max(1, int(np.ceil(float(keep_ratio) * N)))
    idx = np.argsort(score)[-k:]  # top-k

    w = np.ones(N, dtype=np.float64)
    w[idx] = float(w_max)
    return w.astype(np.float32)


# def solve_weights_soft(P, targets, alpha, w_max=8.0, eps=1e-8):
#     N, K = P.shape
#     c = P[:, targets].sum(axis=1)              # [N]
#     c = (c - c.min()) / (c.max() - c.min() + eps)
#     a = float(np.mean(alpha[targets]))         # scalar aggressiveness
#     a = max(0.05, a)
#     w = 1.0 + (w_max - 1.0) * (c ** (1.0 / a))
#     return w.astype(np.float32), {"mode": "soft"}

# def solve_weights_soft(P, targets, alpha, w_max=8.0, eps=1e-8):
#     """
#     Make w depend strongly on target alpha:
#     - compute score c from target influences
#     - keep top-q fraction (q depends on alpha_targets) as high weight, others low weight
#     This yields a sharp, alpha-sensitive weighting and makes GA differences visible.
#     """
#     N, K = P.shape
#     # c = - P[:, targets].sum(axis=1).astype(np.float32)  # [N]
#     target_set = set(targets)
#     non_t = [k for k in range(P.shape[1]) if k not in target_set]
    
#     score_t = (-P[:, targets].sum(axis=1)).astype(np.float32)   # beneficial for targets (after sign fix)
#     score_nt = (P[:, non_t].sum(axis=1)).astype(np.float32)     # proxy of harming others (depends on your P meaning)
    
#     beta = 0.5  # start with 0.5, try 1.0 if still bad
#     c = score_t - beta * score_nt


#     # Normalize scores to [0,1] for stability
#     c = (c - c.min()) / (c.max() - c.min() + eps)

#     # Aggressiveness controlled by alpha on target classes
#     a_t = float(np.mean(alpha[targets]))
#     a_t = np.clip(a_t, 0.0, 2.0)  # since you clip to 2 now

#     # Map alpha -> keep_ratio in (0.1, 0.9)
#     # higher alpha => more selective => smaller keep_ratio => more extreme reweighting
#     # keep_ratio = float(np.clip(0.9 - 0.35 * a_t, 0.10, 0.90))
#     keep_ratio = float(np.clip(0.35 - 0.15 * a_t, 0.05, 0.35))


#     thr = float(np.quantile(c, 1.0 - keep_ratio))  # top keep_ratio gets high weight
#     high = (c >= thr).astype(np.float32)

#     w_low = 1.0
#     w_high = float(w_max)

#     w = w_low + (w_high - w_low) * high  # {1, w_max}
#     return w.astype(np.float32), {"mode": "soft_quantile", "keep_ratio": keep_ratio, "thr": thr}

def solve_weights_soft(P, targets, alpha, w_max=4.0, eps=1e-8):
    """
    Score = target benefit - beta * non-target harm
    Then pick top keep_ratio region, use continuous weights to avoid harsh shifts.
    """
    N, K = P.shape

    target_set = set(targets)
    non_t = [k for k in range(K) if k not in target_set]

    score_t = (-P[:, targets].sum(axis=1)).astype(np.float32)  # beneficial for targets
    score_nt = (P[:, non_t].sum(axis=1)).astype(np.float32) if len(non_t) else 0.0

    beta = 0.5
    c = score_t - beta * score_nt

    # normalize to [0,1]
    c = (c - c.min()) / (c.max() - c.min() + eps)

    a_t = float(np.mean(alpha[targets]))
    a_t = float(np.clip(a_t, 0.0, 2.0))

    keep_ratio = float(np.clip(0.35 - 0.15 * a_t, 0.05, 0.35))
    # keep_ratio = 0.2
    thr = float(np.quantile(c, 1.0 - keep_ratio))

    c_max = float(c.max())
    den = max(c_max - thr, 1e-6)
    w = 1.0 + (w_max - 1.0) * np.clip((c - thr) / den, 0.0, 1.0)
    # continuous weight from 1 to w_max for samples above threshold
    # w = 1.0 + (w_max - 1.0) * np.clip((c - thr) / (1.0 - thr + 1e-8), 0.0, 1.0)

    return w.astype(np.float32), {"mode": "soft_tradeoff", "keep_ratio": keep_ratio, "thr": thr, "beta": beta}
