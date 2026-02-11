# src/influence/cw_influence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadOnlyInfluenceConfig:
    """
    Compute category-wise influence w.r.t. the last linear layer only.
    This is a resource-friendly approximation suitable for Colab.

    We assume model has:
      - a feature extractor part
      - a final classifier head: model.fc (nn.Linear)
    """
    num_classes: int = 10
    ridge: float = 1e-3          # damping term for Hessian stability
    max_train_samples: int | None = None  # optionally subsample train set for speed
    max_val_samples_per_class: int | None = None  # optionally subsample val per class


@torch.no_grad()
def _extract_features_and_labels(
    model: nn.Module,
    loader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: [N, D] (pre-fc features)
      labels: [N]
    Assumption: model has attributes pool and fc like our ResNet9.
    """
    model.eval()

    feats_list = []
    ys_list = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # forward until before fc
        # Works for our ResNet9: everything except final fc
        # We'll reconstruct forward manually to avoid modifying model code.
        z = model.conv1(x)
        z = model.conv2(z)
        z = model.res1(z)
        z = model.conv3(z)
        z = model.conv4(z)
        z = model.res2(z)
        z = model.pool(z).flatten(1)  # [B, D]

        feats_list.append(z.detach().cpu())
        ys_list.append(y.detach().cpu())

    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(ys_list, dim=0)
    return feats, labels


def _grad_wrt_fc(
    feats: torch.Tensor,  # [N, D]
    labels: torch.Tensor, # [N]
    W: torch.Tensor,      # [C, D]
    b: torch.Tensor,      # [C]
) -> torch.Tensor:
    """
    Compute per-sample gradient of CE loss w.r.t. fc parameters (W,b),
    returned as a flattened vector of shape [N, P], where P = C*D + C.

    This uses the standard softmax gradient:
      grad_W = (p - y_onehot)[:, :, None] * feats[:, None, :]
      grad_b = (p - y_onehot)
    """
    logits = feats @ W.t() + b  # [N, C]
    p = F.softmax(logits, dim=1)  # [N, C]
    y_onehot = F.one_hot(labels, num_classes=W.size(0)).float()  # [N, C]
    diff = (p - y_onehot)  # [N, C]

    grad_W = diff.unsqueeze(2) * feats.unsqueeze(1)  # [N, C, D]
    grad_b = diff  # [N, C]

    grad_flat = torch.cat([grad_W.reshape(feats.size(0), -1), grad_b], dim=1)  # [N, P]
    return grad_flat


def compute_cw_influence_head_only(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    cfg: HeadOnlyInfluenceConfig,
) -> Tuple[torch.Tensor, Dict]:
    """
    Returns:
      P_train: [N_train, K] influence matrix (category-wise), where
               P_train[i, k] = - g_val_k^T H^{-1} g_train_i
      meta: dict with shapes and config

    This is a *head-only* approximation: H is Hessian of val loss w.r.t. fc params.
    """
    assert hasattr(model, "fc") and isinstance(model.fc, nn.Linear), "model must have final fc layer"

    # 1) Extract features (pre-fc) and labels for train/val
    Xtr, ytr = _extract_features_and_labels(model, train_loader, device)
    Xva, yva = _extract_features_and_labels(model, val_loader, device)

    if cfg.max_train_samples is not None and Xtr.size(0) > cfg.max_train_samples:
        idx = torch.randperm(Xtr.size(0))[: cfg.max_train_samples]
        Xtr, ytr = Xtr[idx], ytr[idx]

    # Subsample val per class if requested
    if cfg.max_val_samples_per_class is not None:
        kept = []
        for k in range(cfg.num_classes):
            idxk = torch.where(yva == k)[0]
            if idxk.numel() > cfg.max_val_samples_per_class:
                idxk = idxk[torch.randperm(idxk.numel())[: cfg.max_val_samples_per_class]]
            kept.append(idxk)
        kept = torch.cat(kept, dim=0)
        Xva, yva = Xva[kept], yva[kept]

    # 2) Get fc params
    W = model.fc.weight.detach().cpu()
    b = model.fc.bias.detach().cpu()

    C, D = W.size(0), W.size(1)
    P = C * D + C

    # 3) Per-sample grads for train
    Gtr = _grad_wrt_fc(Xtr, ytr, W, b)  # [Ntr, P]

    # 4) Build Hessian approximation H = (1/Nva) sum_i g_i g_i^T + ridge*I
    # Use val gradients per sample (but we need per-class aggregated g_val_k too)
    Gva = _grad_wrt_fc(Xva, yva, W, b)  # [Nva, P]

    # Hessian (P x P). P can be ~ (10*512 +10)=5130 => Hessian ~ 26M floats ~ OK on CPU RAM
    H = (Gva.t() @ Gva) / max(1, Gva.size(0))
    H = H + cfg.ridge * torch.eye(P)

    H_inv = torch.linalg.inv(H)  # [P, P]

    # 5) For each class k: g_val_k = mean grad over val samples of class k
    P_train = torch.zeros((Gtr.size(0), cfg.num_classes), dtype=torch.float32)

    for k in range(cfg.num_classes):
        idxk = torch.where(yva == k)[0]
        if idxk.numel() == 0:
            continue
        gk = Gva[idxk].mean(dim=0, keepdim=True)  # [1, P]
        # influence: - gk H^{-1} Gtr^T  => [1,P] [P,P] [P,Ntr] => [1,Ntr]
        infl = -(gk @ H_inv @ Gtr.t()).squeeze(0)  # [Ntr]
        P_train[:, k] = infl

    meta = {
        "method": "head_only",
        "num_classes": cfg.num_classes,
        "ridge": cfg.ridge,
        "train_N": int(Gtr.size(0)),
        "val_N": int(Gva.size(0)),
        "fc_dim": int(D),
        "param_dim": int(P),
    }
    return P_train, meta
