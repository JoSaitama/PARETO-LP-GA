# src/influence/ekfac_influence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.influence.ekfac import EKFACConfig, build_ekfac_blocks, EKFACBlock


@dataclass
class EKFACInfluenceConfig:
    num_classes: int = 10
    damping: float = 1e-3
    # how many batches to estimate factors (None -> full pass)
    max_factor_batches: Optional[int] = None
    # val samples per class to estimate g_val,k
    max_val_per_class: int = 500
    # per-sample gradient mode
    mode: str = "subsample"          # "full" or "subsample"
    max_train_samples: Optional[int] = 5000
    seed: int = 42
    # choose layers by name substring (e.g., "layer4" or "fc"); None -> all supported layers
    layer_filter: Optional[str] = None


@torch.no_grad()
def _select_val_indices_by_class(val_loader: DataLoader, num_classes: int, max_per_class: int) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Collect a small pool of (x,y) tensors per class from val_loader.
    Stored on CPU; later moved to device in minibatches.
    """
    pools: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {k: [] for k in range(num_classes)}
    counts = [0]*num_classes
    for x, y in val_loader:
        for i in range(x.size(0)):
            k = int(y[i].item())
            if counts[k] < max_per_class:
                pools[k].append((x[i].cpu(), y[i].cpu()))
                counts[k] += 1
        if all(c >= max_per_class for c in counts):
            break
    return pools


def _zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


@torch.no_grad()
def _blocks_grad_vec(blocks: List[EKFACBlock]) -> List[torch.Tensor]:
    """
    Return per-block gradient matrices (Dout x Din) for current backward.
    """
    mats = []
    for b in blocks:
        mats.append(b.layer_grad_matrix().detach().clone())
    return mats


def _apply_inv_to_block_grads(blocks: List[EKFACBlock], g_mats: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    For each block, compute invFisher * gmat.
    """
    out = []
    for b, g in zip(blocks, g_mats):
        out.append(b.inv_fisher_dot(g))
    return out


def _dot_block_mats(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    """
    Sum over blocks: <A_block, B_block> = sum_{l} sum_{ij} A_l[i,j]*B_l[i,j]
    """
    s = 0.0
    for x, y in zip(a, b):
        s += float((x * y).sum().item())
    return s


def compute_ekfac_influence_Ptrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    cfg: EKFACInfluenceConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Returns:
        P_train: torch.Tensor [N, K] on CPU
        meta: dict
    Assumptions:
        - train_loader must have shuffle=False so we can align index order, OR the dataset is indexed externally.
        - For simplest integration: use shuffle=False and rely on dataset order.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)
    model.train()

    ek_cfg = EKFACConfig(damping=cfg.damping, ema_decay=None, device=device)
    blocks = build_ekfac_blocks(model, ek_cfg, layer_filter=cfg.layer_filter)

    # 1) Estimate EKFAC factors A and G
    factor_batches = 0
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        _zero_grads(model)

        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        for b in blocks:
            b.update_factors()

        factor_batches += 1
        if cfg.max_factor_batches is not None and factor_batches >= cfg.max_factor_batches:
            break

    for b in blocks:
        b.compute_eigendecomp()

    # 2) Compute class-conditional val gradients g_val,k
    model.eval()
    pools = _select_val_indices_by_class(val_loader, cfg.num_classes, cfg.max_val_per_class)

    gval_mats_per_class: List[List[torch.Tensor]] = []
    for k in range(cfg.num_classes):
        # accumulate average gradient matrices across selected val samples for class k
        acc = None
        n = 0
        for (x_cpu, y_cpu) in pools[k]:
            x = x_cpu.unsqueeze(0).to(device)
            y = y_cpu.unsqueeze(0).to(device)

            _zero_grads(model)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            gmats = _blocks_grad_vec(blocks)
            if acc is None:
                acc = [g.clone() for g in gmats]
            else:
                for j in range(len(acc)):
                    acc[j] += gmats[j]
            n += 1

        if acc is None or n == 0:
            # fallback: zeros
            acc = [torch.zeros_like(b.layer_grad_matrix()) for b in blocks]
            n = 1

        for j in range(len(acc)):
            acc[j] /= float(n)
        gval_mats_per_class.append(acc)

    # Precompute invF * g_val,k for each class (saves time)
    # inv_gval_per_class: List[List[torch.Tensor]] = []
    # for k in range(cfg.num_classes):
    #     inv_gval_per_class.append(_apply_inv_to_block_grads(blocks, gval_mats_per_class[k]))

    # NOTE: We apply invF on g_train (per-sample) for better numerical stability.
    # So we keep gval_mats_per_class as-is (not preconditioned).


    # 3) Compute P_train (per-sample or subsample)
    model.train()
    # Determine N
    N = len(train_loader.dataset)
    K = cfg.num_classes
    P = torch.zeros((N, K), dtype=torch.float32)

    # choose indices
    if cfg.mode == "full":
        keep = np.arange(N)
    else:
        m = cfg.max_train_samples if cfg.max_train_samples is not None else min(5000, N)
        keep = np.random.choice(N, size=m, replace=False)
        keep.sort()

    keep_set = set(keep.tolist())
    ptr = 0

    # IMPORTANT: requires train_loader shuffle=False to map to global idx by iteration order
    for x, y in train_loader:
        bsz = x.size(0)
        # global indices in dataset order
        batch_indices = np.arange(ptr, ptr + bsz)
        ptr += bsz

        # figure which items in this batch are selected
        mask = [i for i, gi in enumerate(batch_indices) if gi in keep_set]
        if len(mask) == 0:
            continue

        # for correctness, do per-sample backward on selected items
        for bi in mask:
            xi = x[bi:bi+1].to(device)
            yi = y[bi:bi+1].to(device)

            _zero_grads(model)
            logits = model(xi)
            loss = loss_fn(logits, yi)
            loss.backward()

            gtrain = _blocks_grad_vec(blocks)
            
            # Apply invF on train gradient: inv_gtrain = invF(g_train)
            inv_gtrain = _apply_inv_to_block_grads(blocks, gtrain)
            
            # # influence for each class k: - g_train^T invF g_val,k
            # for k in range(K):
            #     score = -_dot_block_mats(gtrain, inv_gval_per_class[k])
            #     P[batch_indices[bi], k] = float(score)
            
            # influence for each class k: - (invF g_train)^T g_val,k
            for k in range(K):
                score = _dot_block_mats(inv_gtrain, gval_mats_per_class[k])
                # score = -_dot_block_mats(inv_gtrain, gval_mats_per_class[k])
                P[batch_indices[bi], k] = float(score)

    # close hooks
    for b in blocks:
        b.close()

    meta = {
        "method": "EKFAC/KFAC influence (layerwise Kronecker factors)",
        "mode": cfg.mode,
        "layer_filter": cfg.layer_filter,
        "damping": cfg.damping,
        "max_factor_batches": cfg.max_factor_batches,
        "max_val_per_class": cfg.max_val_per_class,
        "max_train_samples": cfg.max_train_samples,
        "note": "P_train computed using class-conditional mean val gradients; requires train_loader shuffle=False for index alignment.",
    }
    return P, meta

