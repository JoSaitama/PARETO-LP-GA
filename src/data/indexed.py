# src/data/indexed.py
from __future__ import annotations
from typing import Any, Tuple

class IndexedDataset:
    """
    Wrap a torchvision dataset so that __getitem__ returns (x, y, idx).
    """
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int]:
        x, y = self.base[idx]
        return x, y, idx

    @property
    def targets(self):
        return self.base.targets
