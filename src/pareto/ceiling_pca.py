# src/pareto/ceiling_pca.py
from __future__ import annotations
import numpy as np

def explained_var_ratio_first_pc(P: np.ndarray) -> float:
    """
    P: [N, K] influence matrix.
    We compute EVR of the first principal component using KxK covariance (cheap for K=10).
    """
    P = np.asarray(P, dtype=np.float64)
    P = P - P.mean(axis=0, keepdims=True)
    # covariance: KxK
    cov = (P.T @ P) / max(1, P.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total = float(eigvals.sum()) if eigvals.sum() > 0 else 1.0
    return float(eigvals[0] / total)

