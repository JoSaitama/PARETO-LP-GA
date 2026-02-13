# src/pareto/ga_search.py
from __future__ import annotations
import numpy as np

# def sample_alpha(
#     K: int,
#     target_classes: list[int],
#     low_target: float = 0.6,
#     high_target: float = 1.0,
#     low_nontarget: float = 0.85,
#     high_nontarget: float = 1.0,
#     rng: np.random.Generator | None = None,
# ) -> np.ndarray:
#     """
#     alpha_k in [0,1]. We usually allow more slack on non-target (slightly below 1),
#     and keep target also in a reasonable range.
#     """
#     if rng is None:
#         rng = np.random.default_rng(0)
#     alpha = np.empty(K, dtype=np.float32)
#     for k in range(K):
#         if k in target_classes:
#             alpha[k] = rng.uniform(low_target, high_target)
#         else:
#             alpha[k] = rng.uniform(low_nontarget, high_nontarget)
#     return alpha

import numpy as np

def sample_alpha(K: int, target_classes, rng: np.random.Generator) -> np.ndarray:
    """
    Paper-consistent: alpha_k in [0,1].
    Keep diversity while slightly biasing targets upward.
    """
    a = rng.uniform(0.0, 1.0, size=K).astype(np.float32)

    # bias targets to be larger, but still within [0,1]
    for t in target_classes:
        if 0 <= int(t) < K:
            a[int(t)] = rng.uniform(0.6, 1.0)

    return a

def mutate_alpha(alpha: np.ndarray, sigma: float = 0.05, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    a = alpha + rng.normal(0.0, sigma, size=alpha.shape).astype(np.float32)
    return np.clip(a, 0.0, 1.0)

