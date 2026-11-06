"""Spiking-ResNet-18 backbone (SD^2-SNN substrate).

Replaces CLS-ER's ReLU ResNet-18 with the spiking network validated in the
SD^2-SNN reproduction: ``out = LIF(BN(conv2(LIF(BN(conv1(x))))) + shortcut(x))``,
time-first tensors ``(T, B, C, H, W)`` and BN statistics shared over the flattened
``T*B`` batch.

Interface kept compatible with CLS-ER's usage: ``forward(x) -> logits``, plus
``features`` / ``forward_rates`` for the optional SD^2 regularisers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import LIFNode, RateMonitor


def _flatten_t(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    T, B = x.shape[0], x.shape[1]
    return x.reshape(T * B, *x.shape[2:]), T, B


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, **lif_kw) -> None:
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
        self.lif1 = LIFNode(**lif_kw)
        self.lif2 = LIFNode(**lif_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, T, B = _flatten_t(x)
        y = self.bn1(self.conv1(y))
        y = self.lif1(y.reshape(T, B, *y.shape[1:]))

        y2, _, _ = _flatten_t(y)
        y2 = self.bn2(self.conv2(y2))
        y2 = y2.reshape(T, B, *y2.shape[1:])

        s, _, _ = _flatten_t(x)
        sc = self.shortcut(s)
        sc = sc.reshape(T, B, *sc.shape[1:])
        return self.lif2(y2 + sc)


class SpikingStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, **lif_kw) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNode(**lif_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, T, B = _flatten_t(x)
        y = self.bn(self.conv(y))
        return self.lif(y.reshape(T, B, *y.shape[1:]))


class SpikingResNet18(nn.Module):
    """Spiking ResNet-18 with a time-aggregated (rate) classifier head."""

    def __init__(self, num_classes: int = 100, img_size: int = 32, width: int = 32,
                 tau: float = 2.0, v_threshold: float = 1.0, surrogate_alpha: float = 4.0,
                 detach_reset: bool = True, energy_layer: Optional[nn.Linear] = None,
                 readout: str = "rate", readout_norm: bool = True,
                 learnable_tau: bool = False, learnable_vth: bool = False,
                 kwta_ratio: float = 0.0,
                 kwta_expand: int = 1, fixed_head: str = "none",
                 fixed_head_norm: bool = False, fixed_head_scale: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.width = width
        self.readout = str(readout).lower()
        self.kwta_ratio = float(kwta_ratio or 0.0)
        self.kwta_expand = max(1, int(kwta_expand or 1))
        if self.readout not in ("rate", "membrane"):
            raise ValueError(f"unknown readout {readout!r}")
        lif_kw = dict(tau=tau, v_threshold=v_threshold,
                      surrogate_alpha=surrogate_alpha, detach_reset=detach_reset,
                      learnable_tau=learnable_tau, learnable_vth=learnable_vth)

        self.in_planes = width
        self.stem = SpikingStem(3, width, **lif_kw)
        self.downsample = nn.MaxPool2d(2) if img_size >= 64 else nn.Identity()
        self.layer1 = self._make_layer(width, 2, 1, **lif_kw)
        self.layer2 = self._make_layer(width * 2, 2, 2, **lif_kw)
        self.layer3 = self._make_layer(width * 4, 2, 2, **lif_kw)
        self.layer4 = self._make_layer(width * 8, 2, 2, **lif_kw)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        code_dim = width * 8 * self.kwta_expand
        # dentate-gyrus expansion (EC -> DG): C -> r*C, then ReLU, then k-WTA
        self.dg = (nn.Linear(width * 8, code_dim)
                   if self.kwta_expand > 1 else None)
        self.fc = nn.Linear(code_dim, num_classes)
        # membrane readout is unbounded; keep the scale comparable to spike rates
        self.readout_norm = (nn.LayerNorm(width * 8)
                             if (self.readout == "membrane" and readout_norm)
                             else nn.Identity())

        # ---- PS-SNN style head: fixed, mutually orthogonal class centers -----
        # "Most SNN-CL methods re-initialise the classifier at every step and
        # optimise it on imbalanced data, which drifts the representation." A head
        # that never moves removes that drift entirely: the backbone must align its
        # features to a *fixed* set of orthogonal targets instead of chasing a
        # moving decision boundary (PS-SNN, Sci. Rep. 2026).
        self.fixed_head = str(fixed_head).lower()
        self.fixed_head_norm = bool(fixed_head_norm)
        self.fixed_head_scale = float(fixed_head_scale)
        if self.fixed_head == "orthogonal":
            if num_classes > code_dim:
                raise ValueError(f"cannot place {num_classes} orthogonal centers "
                                 f"in {code_dim} dims")
            nn.init.orthogonal_(self.fc.weight)      # rows orthonormal, norm 1
            with torch.no_grad():
                self.fc.bias.zero_()
            self.fc.weight.requires_grad_(False)
            self.fc.bias.requires_grad_(False)
        elif self.fixed_head != "none":
            raise ValueError(f"unknown fixed_head {fixed_head!r}")

    def _make_layer(self, planes: int, blocks: int, stride: int, **lif_kw) -> nn.Sequential:
        layers: List[nn.Module] = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(SpikingBasicBlock(self.in_planes, planes, s, **lif_kw))
            self.in_planes = planes
        return nn.Sequential(*layers)

    # ------------------------------------------------------------- forward
    def classify(self, code: torch.Tensor) -> torch.Tensor:
        """Apply the (optionally frozen, orthogonal) classifier to a code."""
        if self.fixed_head == "orthogonal":
            if self.fixed_head_norm:
                code = F.normalize(code, dim=-1)
            return self.fixed_head_scale * F.linear(code, self.fc.weight, self.fc.bias)
        return self.fc(code)

    def _run_stages(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if not isinstance(self.downsample, nn.Identity):
            T, B = x.shape[0], x.shape[1]
            x = self.downsample(x.reshape(T * B, *x.shape[2:])).reshape(T, B, *x.shape[2:])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(T, B, 3, H, W) -> pooled features.

        Two readouts, selected by ``self.readout``:

        ``rate``      (T, B, C) — mean spike count per neuron. With T=4 each
                      component takes only 5 distinct values, a very coarse
                      quantisation of the evidence.
        ``membrane``  (1, B, C) — the last layer's membrane potential *before* the
                      final reset: continuous, and it accumulates evidence over the
                      whole window. Costs nothing extra (same T).
        """
        x = self._run_stages(x)
        if self.readout == "membrane":
            last_lif = self.layer4[-1].lif2
            v = last_lif.last_v                    # (B, C', H, W), pre-reset
            pooled = self.avgpool(v).reshape(v.shape[0], -1)
            pooled = self.readout_norm(pooled)     # keep the scale sane for the head
            return pooled.unsqueeze(0)             # (1, B, C)
        T, B = x.shape[0], x.shape[1]
        x = self.avgpool(x.reshape(T * B, *x.shape[2:]))
        return x.reshape(T, B, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)          # (T, B, C)
        rate = self.features_to_code(feats)   # (B, C)  ← dentate-gyrus sparse code
        return self.classify(rate)

    # ------------------------------------------------ dentate gyrus (k-WTA)
    def features_to_code(self, feats: torch.Tensor) -> torch.Tensor:
        """Cortical code -> dentate-gyrus code: **expand, then sparsify**.

        The dentate gyrus performs *pattern separation*: granule cells are an order
        of magnitude more numerous than their cortical inputs (expansion), and only
        ~2–5 % of them are active at a time (sparsity). Together that gives codes
        that barely overlap for overlapping inputs, which is what keeps memories
        from interfering.

        Two knobs reproduce it:

        * ``kwta_expand`` (r): a learned linear expansion ``C -> r·C`` followed by
          ReLU. Without it, sparsifying *in place* throws information away (measured:
          256 -> 13 survivors cost ~1.5 points everywhere).
        * ``kwta_ratio`` (ρ): keep the top ``ρ·r·C`` entries per sample, zero the
          rest. Gradient flows only through the survivors.

        Note the product that matters: with ``r=4, ρ=0.05`` the code keeps
        ``0.05 × 1024 = 51`` active units — *more* than the original 256-dim dense
        code would suggest at "5 %", so information is preserved while the codes
        become sparse and mutually orthogonal.
        """
        rate = feats.mean(dim=0) if feats.shape[0] > 1 else feats[0]
        if self.dg is not None:
            rate = torch.relu(self.dg(rate))
        return self._kwta(rate, self.kwta_ratio)

    @staticmethod
    def _kwta(x: torch.Tensor, ratio: float) -> torch.Tensor:
        """Differentiable top-k mask (gradient flows only through kept entries)."""
        if ratio is None or ratio <= 0.0:
            return x
        k = max(1, int(round(x.shape[-1] * float(ratio))))
        k = min(k, x.shape[-1])
        idx = x.topk(k, dim=-1).indices
        mask = torch.zeros_like(x).scatter_(-1, idx, 1.0)
        return x * mask

    def forward_temporal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, logits_per_timestep, feats)``.

        ``logits_per_timestep`` has shape (T, B, K) and is what the temporal
        distillation term consumes: it says *when* the class evidence appears,
        which an ANN cannot express. ``feats`` (T, B, C) also feeds the
        spike-count reliability weights.
        """
        feats = self.features(x)                     # (T, B, C)
        rate = self.features_to_code(feats)          # (B, code) — k-WTA applied here
        logits = self.classify(rate)                 # (B, K)
        # per-timestep logits must go through the *same* DG pipeline, otherwise the
        # head sees a C-dim input while it expects r*C
        T, B, C = feats.shape
        flat = feats.reshape(-1, C)
        if self.dg is not None:
            flat = torch.relu(self.dg(flat))
        logits_t = self.classify(flat).reshape(T, B, -1)   # (T, B, K)
        return logits, logits_t, feats

    def forward_with_rates(self, x: torch.Tensor,
                           monitor: Optional[RateMonitor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.forward(x)
        rates = monitor.rates if monitor is not None else {}
        return logits, dict(rates)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_backbone(cfg):
    """Build the substrate selected by ``cfg.backbone`` ("spiking" | "ann")."""
    n_classes = getattr(cfg, "num_classes", 0) or (cfg.classes_per_task * cfg.n_tasks)
    kind = str(getattr(cfg, "backbone", "spiking")).lower()
    if kind == "ann":
        from .backbone_ann import DenseResNet18
        return DenseResNet18(num_classes=n_classes, img_size=cfg.img_size,
                             width=cfg.width)
    if kind != "spiking":
        raise ValueError(f"unknown backbone {kind!r}")
    return SpikingResNet18(
        num_classes=n_classes,
        img_size=cfg.img_size,
        width=cfg.width,
        tau=cfg.tau,
        v_threshold=cfg.v_threshold,
        surrogate_alpha=cfg.surrogate_alpha,
        detach_reset=cfg.detach_reset,
        readout=getattr(cfg, "readout", "rate"),
        readout_norm=getattr(cfg, "readout_norm", True),
        learnable_tau=getattr(cfg, "lif_learnable", False),
        learnable_vth=getattr(cfg, "lif_learnable_vth", False),
        kwta_ratio=getattr(cfg, "kwta_ratio", 0.0),
        kwta_expand=getattr(cfg, "kwta_expand", 1),
        fixed_head=getattr(cfg, "fixed_head", "none"),
        fixed_head_norm=getattr(cfg, "fixed_head_norm", False),
        fixed_head_scale=getattr(cfg, "fixed_head_scale", 1.0),
    )
