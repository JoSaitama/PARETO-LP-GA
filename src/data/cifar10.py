from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class CIFAR10Config:
    data_root: str = "/content/drive/MyDrive/cifar10_data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


def get_cifar10_loaders(cfg: CIFAR10Config) -> Tuple[DataLoader, DataLoader]:
    """
    Returns:
        train_loader, test_loader
    Notes:
        - We store dataset under cfg.data_root (recommend Google Drive for persistence).
        - Using standard CIFAR-10 normalization.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=cfg.data_root, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, test_loader

