# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, makes CUDA ops more deterministic but can slow training.

    Notes:
        - For deep learning workloads, full determinism is not always guaranteed.
        - We default to speed-friendly settings (deterministic=False) which are still stable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions, this enforces deterministic algorithms.
            # If unavailable, ignore.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            # Faster, usually fine for CIFAR-10 baseline replication.
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except Exception:
        # Torch not installed or unavailable; ignore.
        pass
