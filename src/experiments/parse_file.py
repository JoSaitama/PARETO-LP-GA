# src/experiments/parse_file.py
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    # Change these once, everything else follows.
    data_root: str = "/content/drive/MyDrive/cifar10_data"
    output_root: str = "/content/drive/MyDrive/pareto_influence_outputs"
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 20
    aug: str = "train_aug"   # "train_aug" or "noaug"


D = Defaults()


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common CLI args with safe defaults."""
    p.add_argument("--data_root", type=str, default=D.data_root)
    p.add_argument("--output_root", type=str, default=D.output_root)

    p.add_argument("--seed", type=int, default=D.seed)
    p.add_argument("--batch_size", type=int, default=D.batch_size)
    p.add_argument("--num_workers", type=int, default=D.num_workers)

    p.add_argument("--epochs", type=int, default=D.epochs)
    p.add_argument("--aug", type=str, default=D.aug, choices=["train_aug", "noaug"])
    return p


def exp_dirs(output_root: str, tag: str, seed: int) -> dict:
    """
    Standard experiment directories.
    Example tag: "resnet9_cifar10"
    """
    baseline = os.path.join(output_root, f"baseline_{tag}_seed{seed}")
    influence = os.path.join(output_root, f"influence_{tag}_seed{seed}")
    retrain = os.path.join(output_root, f"delete_retrain_{tag}_seed{seed}")
    return {"baseline": baseline, "influence": influence, "retrain": retrain}


def ckpt_path(baseline_dir: str) -> str:
    return os.path.join(baseline_dir, "checkpoints", "best.pt")
