"""Dense (non-spiking) ResNet-18 control with the *same* interface as the SNN.

The point of this file is the one control the project was missing: hold everything
else fixed (data, epochs, buffer, CLS-ER mechanics, parameter count) and swap only
the substrate. That separates "how much does spiking cost?" from "how much is the
training budget / hyper-parameters?" — a distinction none of the previous eight
experiments could make.

Interface contract (so ``SpikingCLSER`` runs unchanged):

* ``forward(x)`` with ``x`` of shape (T, B, 3, H, W) -> logits (B, K).
  The time axis is simply averaged over before the first conv (an ANN has no time).
* ``features(x)`` -> (1, B, C), so that ``.mean(dim=0)`` yields (B, C) exactly as
  the spiking backbone does.
* ``forward_temporal(x)`` -> (logits, logits_t, feats) with ``logits_t`` of shape
  (1, B, K); the temporal-distillation option is trivially inactive here.
* ``num_params``.

Parameter count at ``width=32`` is ~2.79 M — the same as the spiking width-32
network (2.80 M), which is what makes the comparison fair.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class DenseResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, img_size: int = 32,
                 width: int = 32, **_ignored) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.width = width

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.MaxPool2d(2) if img_size >= 64 else nn.Identity()
        self.in_planes = width
        self.layer1 = self._make_layer(width, 2, 1)
        self.layer2 = self._make_layer(width * 2, 2, 2)
        self.layer3 = self._make_layer(width * 4, 2, 2)
        self.layer4 = self._make_layer(width * 8, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(_Block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _collapse_time(x: torch.Tensor) -> torch.Tensor:
        """(T, B, 3, H, W) -> (B, 3, H, W) by averaging the frames."""
        return x.mean(dim=0) if x.dim() == 5 else x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self._collapse_time(x)
        h = self.stem(h)
        h = self.downsample(h)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        h = self.avgpool(h).flatten(1)
        return h.unsqueeze(0)                      # (T=1, B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).mean(dim=0))

    def forward_temporal(self, x: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.features(x)                   # (1, B, C)
        logits = self.fc(feats.mean(dim=0))
        return logits, self.fc(feats.reshape(-1, feats.shape[-1])).reshape(
            feats.shape[0], feats.shape[1], -1), feats

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
