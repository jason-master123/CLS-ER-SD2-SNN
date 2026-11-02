"""Optional SD^2-SNN regularisers, kept independent of CLS-ER's core scheme.

Four pieces, ported from the validated SD^2-SNN reproduction:

===========================  ==================================================
Eq.(16)  W ~= U V^T          ``FactorizedConv2d`` / ``FactorizedLinear``
Eq.(17)  eta_U << eta_V      ``parameter_groups`` (U gets ``u_lr_ratio`` x lr)
Eq.(18)  ||U_k - U_{k-1}||^2 ``basis_snapshot`` + ``stability_penalty``
Eq.(13)(14) sparse subspace  ``SparseSubspaceMemory``
Eq.(3)   KL(rho || r_hat)     ``kl_sparsity``
===========================  ==================================================

All of them are off by default: the CLS-ER scheme alone is a complete method, and
these are the SD^2-SNN contributions that can be layered on top.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Eq.(16) structured decomposition:  W ~= U V^T
# --------------------------------------------------------------------------- #
class FactorizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 rank: Optional[int] = None, rank_ratio: float = 0.75) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        full = min(out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.rank = int(rank) if rank is not None else max(1, int(round(rank_ratio * full)))
        self.rank = max(1, min(self.rank, full))
        self.U = nn.Parameter(torch.empty(out_channels, self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        return torch.einsum("or,rikl->oikl", self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, rank_ratio: float = 0.75) -> "FactorizedConv2d":
        m = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding,
                bias=conv.bias is not None, rank_ratio=rank_ratio)
        with torch.no_grad():
            w = conv.weight.detach().reshape(conv.out_channels, -1)
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            r = m.rank
            m.U.copy_(u[:, :r] * s[:r].unsqueeze(0))
            m.V.copy_(vh[:r].reshape(r, *m.V.shape[1:]))
            if conv.bias is not None:
                m.bias.copy_(conv.bias.detach())
        return m


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank: Optional[int] = None, rank_ratio: float = 0.75) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        full = min(in_features, out_features)
        self.rank = int(rank) if rank is not None else max(1, int(round(rank_ratio * full)))
        self.rank = max(1, min(self.rank, full))
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        return self.U @ self.V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank_ratio: float = 0.75) -> "FactorizedLinear":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, rank_ratio=rank_ratio)
        with torch.no_grad():
            w = lin.weight.detach()
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            r = m.rank
            m.U.copy_(u[:, :r] * s[:r].unsqueeze(0))
            m.V.copy_(vh[:r])
            if lin.bias is not None:
                m.bias.copy_(lin.bias.detach())
        return m


def decomposed_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (FactorizedConv2d, FactorizedLinear)):
            yield name, module


def replace_with_factorized(model: nn.Module, rank_ratio: float = 0.75,
                            include_fc: bool = False) -> nn.Module:
    """Swap Conv2d (and optionally Linear) sub-modules in place, SVD-initialised."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, FactorizedConv2d.from_conv(child, rank_ratio))
        elif isinstance(child, nn.Linear) and include_fc:
            setattr(model, name, FactorizedLinear.from_linear(child, rank_ratio))
        else:
            replace_with_factorized(child, rank_ratio, include_fc)
    return model


def parameter_groups(model: nn.Module, lr: float, weight_decay: float,
                     u_lr_ratio: float = 0.1) -> List[dict]:
    """Eq.(17): the shared basis U is updated ``u_lr_ratio`` times slower."""
    u_ids, v_ids = set(), set()
    for _, module in decomposed_layers(model):
        u_ids.add(id(module.U))
        v_ids.add(id(module.V))
    u_params, v_params, other = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in u_ids:
            u_params.append(p)
        elif id(p) in v_ids:
            v_params.append(p)
        else:
            other.append(p)
    groups = [{"params": other, "lr": lr, "weight_decay": weight_decay, "name": "base"}]
    if v_params:
        groups.append({"params": v_params, "lr": lr, "weight_decay": weight_decay, "name": "V"})
    if u_params:
        groups.append({"params": u_params, "lr": lr * u_lr_ratio,
                       "weight_decay": weight_decay, "name": "U"})
    return groups


def basis_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: m.U.detach().clone() for name, m in decomposed_layers(model)}


def stability_penalty(model: nn.Module, prev: Optional[dict]) -> torch.Tensor:
    """Eq.(18): sum_l ||U_k^l - U_{k-1}^l||^2."""
    if not prev:
        return torch.zeros((), device=next(model.parameters()).device)
    terms = []
    for name, module in decomposed_layers(model):
        ref = prev.get(name)
        if ref is not None and ref.shape == module.U.shape:
            terms.append((module.U - ref.to(module.U.device)).pow(2).sum())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).sum()


# --------------------------------------------------------------------------- #
#  Eq.(3) temporal sparsity, Eq.(13)-(14) sparse subspace memory
# --------------------------------------------------------------------------- #
def kl_sparsity(rates: torch.Tensor, rho: float = 0.05, eps: float = 1e-6) -> torch.Tensor:
    """KL(rho || r_hat) per neuron, averaged over the batch (Eq. 2/3)."""
    p = rates.float().clamp(eps, 1.0 - eps)
    rho_t = torch.as_tensor(rho, dtype=p.dtype, device=p.device)
    return rho_t * torch.log(rho_t / p) + (1 - rho_t) * torch.log((1 - rho_t) / (1 - p))


def layer_sparsity(rate_dict: Dict[str, torch.Tensor], rho: float = 0.05) -> torch.Tensor:
    if not rate_dict:
        return torch.zeros(())
    terms = [kl_sparsity(r, rho).mean() for r in rate_dict.values()]
    return torch.stack(terms).mean()


class SparseSubspaceMemory:
    """Per-task set of important neurons + their reference firing rates."""

    def __init__(self, eps: float = 0.02, normalize: bool = True) -> None:
        self.eps = eps
        self.normalize = normalize
        self.entries: List[dict] = []

    def update(self, task: int, mean_rates: Dict[str, torch.Tensor]) -> None:
        entry = {"task": task, "idx": {}, "rate": {}}
        for name, r in mean_rates.items():
            r = r.detach().float().cpu()
            idx = torch.nonzero(r > self.eps, as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = torch.topk(r, k=min(8, r.numel())).indices
            entry["idx"][name] = torch.sort(idx).values
            entry["rate"][name] = r[entry["idx"][name]]
        self.entries.append(entry)

    def loss(self, current: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.entries or not current:
            device = next(iter(current.values())).device if current else "cpu"
            return torch.zeros((), device=device)
        device = next(iter(current.values())).device
        total, n = torch.zeros((), device=device), 0
        for entry in self.entries:
            for name, idx in entry["idx"].items():
                if name not in current:
                    continue
                r_cur = current[name].mean(dim=0).float()
                idx = idx.to(r_cur.device)
                if idx.numel() == 0 or idx.max() >= r_cur.numel():
                    continue
                diff = (r_cur[idx] - entry["rate"][name].to(r_cur.device)).pow(2).sum()
                total = total + (diff / idx.numel() if self.normalize else diff)
                n += 1
        return total / max(1, n)

    def n_stored(self) -> int:
        return int(sum(v.numel() for e in self.entries for v in e["idx"].values()))

    def summary(self) -> dict:
        return {"tasks": len(self.entries), "stored_neurons": self.n_stored(),
                "per_task": [int(sum(v.numel() for v in e["idx"].values())) for e in self.entries]}
