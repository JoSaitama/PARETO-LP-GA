from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Residual(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet9(nn.Module):
    """
    Lightweight ResNet-9 style network for CIFAR-10.
    Produces logits of shape [B, num_classes].
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 32x32
        self.conv1 = ConvBNReLU(3, 64)
        self.conv2 = ConvBNReLU(64, 128, s=2)  # 16x16
        self.res1 = Residual(nn.Sequential(
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
        ))

        self.conv3 = ConvBNReLU(128, 256, s=2)  # 8x8
        self.conv4 = ConvBNReLU(256, 512, s=2)  # 4x4
        self.res2 = Residual(nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
        ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

