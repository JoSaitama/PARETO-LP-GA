# src/influence/ekfac.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EKFACConfig:
    """
    EKFAC / KFAC approximation settings.

    damping: numerical stability term added to eigenvalues (or kron eigenvalues).
    ema_decay: if you want EMA updates of factors; if None, use simple average.
    """
    damping: float = 1e-3
    ema_decay: Optional[float] = None
    device: str = "cuda"


def _is_supported_layer(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, nn.Conv2d))


def _flatten_conv_input(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """
    For conv layers, KFAC uses unfolded patches as 'input activations' A.
    Return shape: [B, Cin*Kh*Kw (+1 if bias)].
    """
    # x: [B, Cin, H, W]
    unfold = F.unfold(
        x,
        kernel_size=conv.kernel_size,
        dilation=conv.dilation,
        padding=conv.padding,
        stride=conv.stride,
    )  # [B, Cin*Kh*Kw, L]
    # Treat each spatial location as an independent sample
    # reshape to [B*L, Cin*Kh*Kw]
    B, D, L = unfold.shape
    a = unfold.transpose(1, 2).contiguous().view(B * L, D)
    return a


def _layer_input_matrix(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """
    Return activation matrix A_input for EKFAC:
      - Linear: [B, in_features (+1 if bias)]
      - Conv2d: [B*L, Cin*Kh*Kw (+1 if bias)]
    """
    if isinstance(layer, nn.Linear):
        a = x.view(x.size(0), -1)
    elif isinstance(layer, nn.Conv2d):
        a = _flatten_conv_input(x, layer)
    else:
        raise TypeError(f"Unsupported layer: {type(layer)}")

    if layer.bias is not None:
        ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
        a = torch.cat([a, ones], dim=1)
    return a


def _layer_grad_output_matrix(grad_out: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """
    Return pre-activation gradient matrix G_output for EKFAC:
      - Linear: grad_out is [B, out_features] -> [B, out_features]
      - Conv2d: grad_out is [B, Cout, Hout, Wout] -> [B*L, Cout]
    """
    if isinstance(layer, nn.Linear):
        g = grad_out.view(grad_out.size(0), -1)
    elif isinstance(layer, nn.Conv2d):
        # [B, Cout, H, W] -> [B*L, Cout]
        B, C, H, W = grad_out.shape
        g = grad_out.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
    else:
        raise TypeError(f"Unsupported layer: {type(layer)}")
    return g


class EKFACBlock:
    """
    Store EKFAC factors (A and G), their eigendecomposition, and apply inverse-approx.
    """

    def __init__(self, layer: nn.Module, name: str, cfg: EKFACConfig):
        if not _is_supported_layer(layer):
            raise TypeError(f"Unsupported layer for EKFAC: {type(layer)}")
        self.layer = layer
        self.name = name
        self.cfg = cfg

        self.A: Optional[torch.Tensor] = None  # [Din, Din]
        self.G: Optional[torch.Tensor] = None  # [Dout, Dout]
        self.countA: int = 0
        self.countG: int = 0

        self.QA: Optional[torch.Tensor] = None
        self.LA: Optional[torch.Tensor] = None
        self.QG: Optional[torch.Tensor] = None
        self.LG: Optional[torch.Tensor] = None

        self._x_in: Optional[torch.Tensor] = None
        self._g_out: Optional[torch.Tensor] = None

        # hooks
        self._fh = layer.register_forward_hook(self._forward_hook)
        self._bh = layer.register_full_backward_hook(self._backward_hook)

    def close(self):
        self._fh.remove()
        self._bh.remove()

    def _forward_hook(self, layer, inp, out):
        # inp is a tuple, take first
        x = inp[0].detach()
        self._x_in = x

    def _backward_hook(self, layer, grad_in, grad_out):
        # grad_out is a tuple, take first
        g = grad_out[0].detach()
        self._g_out = g

    @torch.no_grad()
    def update_factors(self):
        """
        Update A and G using stored x_in and g_out from the last forward/backward.
        Call this after backprop on a batch.
        """
        if self._x_in is None or self._g_out is None:
            return

        x = self._x_in
        g = self._g_out

        a = _layer_input_matrix(x, self.layer)          # [M, Din]
        gg = _layer_grad_output_matrix(g, self.layer)   # [M, Dout] (M may be B or B*L)

        # compute second moment
        # A = E[a a^T], G = E[g g^T]
        A_batch = (a.t() @ a) / max(1, a.size(0))
        G_batch = (gg.t() @ gg) / max(1, gg.size(0))

        if self.cfg.ema_decay is None:
            if self.A is None:
                self.A = A_batch
            else:
                self.A = (self.A * self.countA + A_batch) / (self.countA + 1)
            self.countA += 1

            if self.G is None:
                self.G = G_batch
            else:
                self.G = (self.G * self.countG + G_batch) / (self.countG + 1)
            self.countG += 1
        else:
            d = self.cfg.ema_decay
            if self.A is None:
                self.A = A_batch
            else:
                self.A = d * self.A + (1 - d) * A_batch

            if self.G is None:
                self.G = G_batch
            else:
                self.G = d * self.G + (1 - d) * G_batch

        # clear cache
        self._x_in = None
        self._g_out = None

    @torch.no_grad()
    def compute_eigendecomp(self):
        """
        Eigendecompose A and G for EKFAC.
        """
        assert self.A is not None and self.G is not None, f"Factors not ready for {self.name}"
        A = self.A.to(self.cfg.device)
        G = self.G.to(self.cfg.device)

        # symmetric eigendecomp
        LA, QA = torch.linalg.eigh(A)
        LG, QG = torch.linalg.eigh(G)

        # sort descending (optional)
        LA, idxA = torch.sort(LA, descending=True)
        QA = QA[:, idxA]
        LG, idxG = torch.sort(LG, descending=True)
        QG = QG[:, idxG]

        self.QA, self.LA, self.QG, self.LG = QA, LA, QG, LG

    @torch.no_grad()
    def layer_grad_matrix(self) -> torch.Tensor:
        """
        Convert parameter gradients of this layer into a matrix Wgrad of shape [Dout, Din].
        Din includes bias column if bias exists.
        """
        if isinstance(self.layer, nn.Linear):
            gw = self.layer.weight.grad  # [Dout, Din_no_bias]
            if self.layer.bias is not None:
                gb = self.layer.bias.grad.view(-1, 1)  # [Dout, 1]
                gmat = torch.cat([gw, gb], dim=1)
            else:
                gmat = gw
        elif isinstance(self.layer, nn.Conv2d):
            # weight: [Cout, Cin, Kh, Kw] -> [Cout, Cin*Kh*Kw]
            gw = self.layer.weight.grad.view(self.layer.weight.size(0), -1)
            if self.layer.bias is not None:
                gb = self.layer.bias.grad.view(-1, 1)
                gmat = torch.cat([gw, gb], dim=1)
            else:
                gmat = gw
        else:
            raise TypeError(f"Unsupported layer: {type(self.layer)}")
        return gmat

    @torch.no_grad()
    def inv_fisher_dot(self, gmat: torch.Tensor) -> torch.Tensor:
        """
        Apply EKFAC inverse approximation to gradient matrix gmat: returns H^{-1} g (in matrix form).
        gmat: [Dout, Din]
        """
        assert self.QA is not None and self.LA is not None and self.QG is not None and self.LG is not None, \
            f"Eigendecomp not computed for {self.name}"

        QA, LA, QG, LG = self.QA, self.LA, self.QG, self.LG
        damp = self.cfg.damping

        # Project into eigenbasis: g' = QG^T g QA
        gp = (QG.t() @ gmat @ QA)  # [Dout, Din]

        # Scale by inverse kron eigenvalues: 1 / (lg_i * la_j + damp)
        denom = (LG.view(-1, 1) * LA.view(1, -1)) + damp
        gp = gp / denom

        # Project back: g_inv = QG g' QA^T
        ginv = (QG @ gp @ QA.t())
        return ginv


def build_ekfac_blocks(model: nn.Module, cfg: EKFACConfig, layer_filter: Optional[str] = None) -> List[EKFACBlock]:
    """
    Create EKFAC blocks for supported layers.
    layer_filter: if not None, only include layers whose name contains this substring.
    """
    blocks: List[EKFACBlock] = []
    for name, m in model.named_modules():
        if _is_supported_layer(m):
            if layer_filter is None or layer_filter in name:
                blocks.append(EKFACBlock(m, name=name, cfg=cfg))
    return blocks
