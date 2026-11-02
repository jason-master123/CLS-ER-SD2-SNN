"""Logit calibration for class-incremental learning.

The experiments in this repository show the failure mode clearly: after five
tasks the SNN reaches **task-IL 89.0** but only **class-IL 57.5**. Given the task
identity it almost always picks the right class; asked to compare across tasks it
loses. That is a *decision-boundary* problem, not a representation problem, and
post-hoc calibration is the standard cure.

Four components are implemented. All of them are **post-hoc logit transforms**
``z -> a ⊙ z + b``, so they never mutate the network (and therefore never desync
the two EMA memories):

===========================  ==================================================
WA    weight aligning        ``γ = ‖W_new‖ / ‖W_old‖`` (Zhao et al., 2020)
LA    logit adjustment       ``b_i = τ·log π_i`` (Menon et al., 2020) or the
                             balanced-mean variant (see note below)
BiC   bias correction        ``z_old -> α·z_old + β`` (Wu et al., 2019),
                             fitted on the episodic buffer
Cos   cosine classifier      ``z = s · <x̂, ŵ>``, equalises class-norm drift
                             (Gidaris & Komodakis, 2018) — model-side, not post-hoc
===========================  ==================================================

Note on LA: in a balanced class-incremental stream every *seen* class has the
same prior, so the prior term ``τ·log π`` is a constant and cannot separate
tasks. The variant that actually applies to CIL is the ``mean`` mode, which
subtracts the per-class mean logit measured on balanced exemplars — that is the
component which removes "recent classes score higher".

Why logit transforms are equivalent to weight rescaling: ``z = W x`` and scaling
row ``i`` of ``W`` by ``γ_i`` is the same function as scaling ``z_i`` by ``γ_i``.
Doing it in logit space leaves the backbone and the memories untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Cosine classifier (model-side)
# --------------------------------------------------------------------------- #
class CosineHead(nn.Module):
    """``z = scale · <x̂, ŵ> + b``: removes the class-wise norm drift of a linear head.

    Initialised from a trained ``nn.Linear`` so the switch is (nearly) function
    preserving. Note that with a cosine head all class weights have unit norm, so
    weight aligning becomes a no-op by construction — pick one or the other.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 scale: float = 10.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())
        nn.init.xavier_uniform_(self.weight)

    @classmethod
    def from_linear(cls, lin: nn.Linear, scale: float = 10.0) -> "CosineHead":
        head = cls(lin.in_features, lin.out_features, lin.bias is not None, scale)
        with torch.no_grad():
            head.weight.copy_(lin.weight.detach())
            if lin.bias is not None and head.bias is not None:
                head.bias.copy_(lin.bias.detach())
        return head

    @property
    def scaled_weight(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=1) * self.log_scale.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        z = z * self.log_scale.exp()
        return z + self.bias if self.bias is not None else z


def install_cosine_head(model, scale: float = 10.0) -> bool:
    """Replace ``model.net.fc`` with a :class:`CosineHead` (idempotent)."""
    fc = getattr(model.net, "fc", None)
    if fc is None or isinstance(fc, CosineHead):
        return False
    model.net.fc = CosineHead.from_linear(fc, scale=scale)
    return True


# --------------------------------------------------------------------------- #
#  Post-hoc calibrator
# --------------------------------------------------------------------------- #
@dataclass
class LogitCalibrator:
    """``z -> (α·scale)·z + (β + bias)`` on the old classes, ``scale·z + bias`` on the new ones."""

    n_classes: int = 0
    scale: Optional[torch.Tensor] = None      # (K,) per-class multiplicative term
    bias: Optional[torch.Tensor] = None       # (K,) per-class additive term
    alpha: float = 1.0                        # BiC multiplier (old classes only)
    beta: float = 0.0                         # BiC offset     (old classes only)
    old_mask: Optional[torch.Tensor] = None   # (K,) bool
    info: Dict[str, float] = field(default_factory=dict)

    def is_identity(self) -> bool:
        if self.scale is None and self.bias is None:
            return True
        scale_ok = self.scale is None or torch.allclose(
            self.scale, torch.ones_like(self.scale), atol=1e-6)
        bias_ok = self.bias is None or torch.allclose(
            self.bias, torch.zeros_like(self.bias), atol=1e-6)
        return scale_ok and bias_ok and abs(self.alpha - 1.0) < 1e-6 and abs(self.beta) < 1e-6

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        z = logits
        if self.scale is not None:
            z = z * self.scale.to(z.device, z.dtype)
        if self.bias is not None:
            z = z + self.bias.to(z.device, z.dtype)
        if (abs(self.alpha - 1.0) > 1e-6 or abs(self.beta) > 1e-6) and self.old_mask is not None:
            old = self.old_mask.to(z.device).view(1, -1)
            if bool(old.any()):
                # out-of-place: keeps the transform differentiable and never
                # mutates the caller's tensor
                z = torch.where(old, self.alpha * z + self.beta, z)
        return z

    def state_dict(self) -> dict:
        return {"n_classes": self.n_classes,
                "scale": None if self.scale is None else self.scale.detach().cpu(),
                "bias": None if self.bias is None else self.bias.detach().cpu(),
                "alpha": self.alpha, "beta": self.beta,
                "old_mask": None if self.old_mask is None else self.old_mask.detach().cpu(),
                "info": dict(self.info)}

    @classmethod
    def from_state(cls, sd: dict) -> "LogitCalibrator":
        return cls(n_classes=sd["n_classes"], scale=sd["scale"], bias=sd["bias"],
                   alpha=sd["alpha"], beta=sd["beta"], old_mask=sd["old_mask"],
                   info=dict(sd.get("info", {})))

    def summary(self) -> str:
        if self.is_identity():
            return "identity"
        parts = []
        if self.scale is not None:
            parts.append(f"scale∈[{self.scale.min():.2f},{self.scale.max():.2f}]")
        if self.bias is not None:
            parts.append(f"|bias|max={self.bias.abs().max():.2f}")
        if abs(self.alpha - 1.0) > 1e-6 or abs(self.beta) > 1e-6:
            parts.append(f"α={self.alpha:.3f}, β={self.beta:.3f}")
        return ", ".join(parts)


# --------------------------------------------------------------------------- #
#  Fitting helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def class_weight_norms(model) -> torch.Tensor:
    """Effective per-class weight norms of the classifier.

    A :class:`CosineHead` normalises its rows inside ``forward``, so every class
    has unit effective norm and weight aligning is a no-op by construction.
    """
    fc = model.net.fc
    if isinstance(fc, CosineHead):
        return torch.ones(fc.out_features, dtype=torch.float32)
    return fc.weight.detach().norm(dim=1).float()


def fit_weight_alignment(model, old_classes: Sequence[int],
                         new_classes: Sequence[int],
                         clamp: Tuple[float, float] = (0.5, 2.0)
                         ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """WA: bring the old classes' weight norms up to the new classes' level."""
    norms = class_weight_norms(model)
    n_classes = norms.numel()
    scale = torch.ones(n_classes, dtype=torch.float32)
    if not old_classes or not new_classes:
        return scale, {"gamma": 1.0, "gamma_used": 1.0,
                       "norm_old": 0.0, "norm_new": 0.0, "clamped": 0.0}
    old = torch.as_tensor(list(old_classes), dtype=torch.long)
    new = torch.as_tensor(list(new_classes), dtype=torch.long)
    n_old = norms[old].mean()
    n_new = norms[new].mean()
    gamma = float(n_new / n_old.clamp_min(1e-8))
    gamma_c = max(clamp[0], min(clamp[1], gamma))
    scale[old] = gamma_c
    return scale, {"gamma": gamma, "gamma_used": gamma_c,
                   "norm_old": float(n_old), "norm_new": float(n_new),
                   "clamped": float(abs(gamma - gamma_c) > 1e-9)}


@torch.no_grad()
def fit_logit_adjustment(model, examples: torch.Tensor, labels: torch.Tensor,
                         seen_classes: Sequence[int], mode: str = "mean",
                         tau: float = 1.0, batch_size: int = 64,
                         encode_fn=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """LA: per-class additive correction measured on exemplars.

    ``mean``  — ``b_c = E_balanced[ z_c(x) ]``, i.e. the **marginal** mean of class
                ``c``'s logit over all exemplars (NOT only the exemplars labelled
                ``c``). Subtracting it equalises how large each class's score is on
                a balanced stream, which is exactly the class-incremental bias.
                Using the own-class average instead measures "self-confidence",
                which is largest for the most recently trained classes and would
                therefore *penalise the newest task* — the opposite of what is
                needed. (This estimator was wrong in an earlier version and the
                empirical signature was unmistakable: old-task accuracy dropped by
                10 points while the new-task accuracy rose.)
    ``prior`` — ``b_c = τ·log π_c`` with ``π`` the exemplar class frequencies.
                Note that on a balanced stream this is a constant and has no effect;
                it only matters for long-tailed streams.
    """
    n_classes = int(model.net.fc.out_features)
    bias = torch.zeros(n_classes, dtype=torch.float32)
    seen = torch.as_tensor(list(seen_classes), dtype=torch.long)
    if labels.numel() == 0:
        return bias, {"fitted": 0.0}

    if mode == "prior":
        counts = torch.bincount(labels.cpu(), minlength=n_classes).float()
        pi = (counts[seen] / counts[seen].sum().clamp_min(1e-8)).clamp_min(1e-6)
        bias[seen] = tau * pi.log()
        return bias, {"la_mode": 1.0}

    # ``mean`` mode: marginal per-class mean logit over the whole balanced buffer
    sums = torch.zeros(n_classes)
    n_samples = 0
    for start in range(0, examples.shape[0], batch_size):
        xb = examples[start:start + batch_size]
        z = model.net(encode_fn(xb) if encode_fn is not None else xb)
        z = z.detach().float().cpu()
        sums += z.sum(dim=0)                     # accumulate every class's column
        n_samples += z.shape[0]
    if n_samples == 0:
        return bias, {"fitted": 0.0}
    mean_logits = sums / n_samples
    present = torch.zeros(n_classes, dtype=torch.bool)
    present[seen] = True
    # centre so the average logit magnitude is unchanged by the correction
    shift = mean_logits[present].mean()
    bias[present] = mean_logits[present] - shift
    spread = float(bias[present].abs().max())
    # share of the most frequent prediction on the balanced buffer: 1/n_seen is ideal
    return bias, {"la_spread": spread,
                  "la_uniform_ratio": float(1.0 / max(1, int(present.sum())))}


def fit_bias_correction(model, examples: torch.Tensor, labels: torch.Tensor,
                        seen_classes: Sequence[int], old_classes: Sequence[int],
                        prior: Optional[LogitCalibrator] = None,
                        steps: int = 150, lr: float = 0.01, batch_size: int = 32,
                        encode_fn=None, device: str = "cpu",
                        seed: int = 0) -> Tuple[float, float, Dict[str, float]]:
    """BiC: learn ``α, β`` for the old classes on balanced old/new exemplars."""
    if not old_classes:
        return 1.0, 0.0, {"bic_skipped": 1.0}
    old = torch.as_tensor(list(old_classes), dtype=torch.long)
    labels_cpu = labels.cpu()
    old_idx = torch.nonzero(torch.isin(labels_cpu, old)).flatten()
    new_idx = torch.nonzero(~torch.isin(labels_cpu, old)).flatten()
    if old_idx.numel() < 4 or new_idx.numel() < 4:
        return 1.0, 0.0, {"bic_skipped": 1.0}

    params = torch.zeros(2, device=device, requires_grad=True)     # (log α, β)
    params.data[0] = 0.0
    opt = torch.optim.Adam([params], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    seen = torch.as_tensor(list(seen_classes), dtype=torch.long, device=device)
    n_half = max(4, batch_size // 2)
    first_loss = last_loss = None

    for _ in range(steps):
        take_old = old_idx[torch.randint(0, old_idx.numel(), (n_half,), generator=gen)]
        take_new = new_idx[torch.randint(0, new_idx.numel(), (n_half,), generator=gen)]
        idx = torch.cat([take_old, take_new])
        xb = examples[idx].to(device)
        yb = labels[idx].to(device)
        with torch.no_grad():
            z = model.net(encode_fn(xb) if encode_fn is not None else xb).float()
            if prior is not None:
                z = prior.transform(z)
        # ``enable_grad`` is required because the caller (``end_of_task``) runs
        # under ``no_grad``; the backbone forward above stays graph-free on purpose
        with torch.enable_grad():
            alpha = params[0].exp()
            beta = params[1]
            # out-of-place: ``z`` carries no graph, so in-place column assignment
            # would sever the one created here
            old_mask = torch.zeros(1, z.shape[1], dtype=torch.bool, device=device)
            old_mask[0, old.to(device)] = True
            z_cal = torch.where(old_mask, alpha * z + beta, z)
            loss = F.cross_entropy(z_cal.index_select(1, seen), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())

    return float(params[0].exp().detach()), float(params[1].detach()), {
        "bic_loss_first": first_loss if first_loss is not None else float("nan"),
        "bic_loss_last": last_loss if last_loss is not None else float("nan")}


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def build_calibrator(cfg, model, buffer, seen_classes: Sequence[int],
                     task_classes: Sequence[int], encode_fn=None,
                     augment_fn=None) -> LogitCalibrator:
    """Fit the calibrators requested by ``cfg.calib_mode`` at a task boundary.

    ``calib_mode`` is a comma separated list of ``wa``, ``la``, ``bic`` (or ``all``).
    """
    modes = {m.strip().lower() for m in str(getattr(cfg, "calib_mode", "none")).split(",")}
    if "all" in modes:
        modes = {"wa", "la", "bic"}
    modes = {m for m in modes if m in {"wa", "la", "bic"}}
    cal = LogitCalibrator(n_classes=int(model.net.fc.out_features))
    if not modes or buffer.is_empty():
        return cal

    old_classes = [c for c in seen_classes if c not in set(task_classes)]
    examples, labels = buffer.get_all_data()
    examples = examples.detach()
    labels = labels.detach()

    if "wa" in modes:
        cal.scale, info = fit_weight_alignment(model, old_classes, list(task_classes))
        cal.info.update({f"wa_{k}": v for k, v in info.items()})
        if info["gamma_used"] == 1.0 and abs(info.get("gamma", 1.0) - 1.0) > 0.05:
            print(f"  [calib] WA γ={info['gamma']:.2f} clamped to 1.0")

    if "la" in modes:
        cal.bias, info = fit_logit_adjustment(
            model, examples, labels, seen_classes,
            mode=getattr(cfg, "calib_la_mode", "mean"),
            tau=float(getattr(cfg, "calib_tau", 1.0)), encode_fn=encode_fn)
        cal.info.update({f"la_{k}": v for k, v in info.items()})

    if "bic" in modes:
        cal.old_mask = torch.zeros(cal.n_classes, dtype=torch.bool)
        if old_classes:
            cal.old_mask[torch.as_tensor(old_classes, dtype=torch.long)] = True
        cal.alpha, cal.beta, info = fit_bias_correction(
            model, examples, labels, seen_classes, old_classes, prior=cal,
            steps=int(getattr(cfg, "bic_steps", 150)),
            lr=float(getattr(cfg, "bic_lr", 0.01)),
            batch_size=int(getattr(cfg, "batch_size", 32)),
            encode_fn=encode_fn, device=str(model.device), seed=int(getattr(cfg, "seed", 0)))
        cal.info.update({f"bic_{k}": v for k, v in info.items()})

    return cal
