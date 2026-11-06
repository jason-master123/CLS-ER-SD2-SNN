"""Leaky Integrate-and-Fire neuron with an ATan surrogate gradient.

Taken from the validated SD^2-SNN implementation (same dynamics as SpikingJelly's
``LIFNode``, which the SD^2-SNN paper used):

    V[t] = V[t-1] + (X[t] - (V[t-1] - V_reset)) / tau
    S[t] = Theta(V[t] - u_th)
    V[t] = V_reset                       where S[t] = 1

The multi-step loop lives inside ``forward`` so the module is stateless between
calls: the whole (T, B, ...) tensor is consumed at once and no global ``reset_net``
bookkeeping is needed.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ATan(torch.autograd.Function):
    """ATan surrogate: g(x)=1/pi*atan(pi/2*a*x)+1/2, g'(x)=(a/2)/(1+(pi/2*a*x)^2)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        grad = grad_output * (a / 2.0) / (1.0 + (math.pi / 2.0 * a * x) ** 2)
        return grad, None


def spike_function(v: torch.Tensor, v_threshold: float, alpha: float = 4.0) -> torch.Tensor:
    return _ATan.apply(v - v_threshold, alpha)


class LIFNode(nn.Module):
    """Leaky integrate-and-fire neuron (optionally *parametric*, i.e. PLIF).

    With ``learnable_tau=True`` the per-layer membrane time constant becomes a
    trainable scalar, written the way PLIF does it:

        v_t = (1 - 1/tau) * v_{t-1} + x_t ,      1/tau = sigmoid(a)

    (Fang et al., ICCV 2021, "Incorporating Learnable Membrane Time Constant").
    Everything else — the ATan surrogate, the hard reset, the stateless multi-step
    loop — is unchanged, so this is a drop-in upgrade of the backbone rather than a
    different network.
    """

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, v_reset: float = 0.0,
                 surrogate_alpha: float = 4.0, detach_reset: bool = True,
                 decay_input: bool = True, learnable_tau: bool = False,
                 learnable_vth: bool = False) -> None:
        super().__init__()
        self.tau = float(tau)
        self._v_th = float(v_threshold)
        self.v_reset = float(v_reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.detach_reset = detach_reset
        self.learnable_tau = bool(learnable_tau)
        self.learnable_vth = bool(learnable_vth)
        # PLIF writes the input term undivided, which corresponds to decay_input=False
        self.decay_input = False if self.learnable_tau else decay_input
        self.last_spike: Optional[torch.Tensor] = None
        if self.learnable_tau:
            # init so that sigmoid(a) = 1/tau  (tau=2 -> a=0)
            init = math.log(1.0 / max(1e-3, min(1.0 - 1e-3, 1.0 / self.tau)))
            init = math.log((1.0 / self.tau) / (1.0 - 1.0 / self.tau))
            self.a = nn.Parameter(torch.tensor(float(init)))
        if self.learnable_vth:
            # v_th = softplus(a_th), initialised to the fixed value (default 1.0)
            v0 = max(1e-3, self._v_th)
            self.a_th = nn.Parameter(torch.tensor(float(math.log(math.expm1(v0)))))

    @property
    def inv_tau(self):
        """1/tau — a tensor when parametric, a float otherwise."""
        return torch.sigmoid(self.a) if self.learnable_tau else (1.0 / self.tau)

    @property
    def v_threshold(self):
        """Firing threshold — a tensor when learnable, a float otherwise.

        Exposed as a property so that everything that used to read/write the plain
        attribute (homeostatic calibration, the full-state EMA of the memories)
        keeps working without knowing about the parametrisation.
        """
        if self.learnable_vth:
            return F.softplus(self.a_th) + 1e-6
        return self._v_th

    @v_threshold.setter
    def v_threshold(self, value) -> None:
        v = float(value)
        if getattr(self, "learnable_vth", False):
            v = max(1e-3, v)
            with torch.no_grad():
                self.a_th.fill_(float(math.log(math.expm1(v))))
        else:
            self._v_th = v

    def extra_repr(self) -> str:
        tau = f"1/tau~sigmoid(a) (init {1.0 / self.tau:.3f})" if self.learnable_tau \
            else f"tau={self.tau}"
        return f"{tau}, v_th={self.v_threshold}, alpha={self.surrogate_alpha}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() <= 4
        if squeeze:
            x = x.unsqueeze(0)
        T = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes: List[torch.Tensor] = []
        v_pre_reset = v
        inv_tau = self.inv_tau                  # tensor (PLIF) or float (fixed tau)
        for t in range(T):
            if self.decay_input:
                v = v + (x[t] - (v - self.v_reset)) / self.tau
            elif self.learnable_tau:
                # PLIF: v_t = (1 - 1/tau) v_{t-1} + x_t
                v = (1.0 - inv_tau) * v + x[t]
            else:
                v = v + x[t] + (self.v_reset - v) / self.tau
            v_pre_reset = v
            s = spike_function(v, self.v_threshold, self.surrogate_alpha)
            # hard reset, optionally detached (standard trick to avoid a dead gradient)
            v = v - (s.detach() if self.detach_reset else s) * (v - self.v_reset)
            spikes.append(s)
        out = torch.stack(spikes, dim=0)
        self.last_spike = out.detach()
        # membrane *before* the final reset: a continuous quantity that keeps the
        # evidence accumulated over the whole window (see ``readout="membrane"``).
        # NOT detached: it is the classifier's input, so the backbone must receive
        # gradient through it (detaching here silently reduces the net to a linear
        # probe on random features — task 0 dropped from 96.1 to 64.8).
        self.last_v = v_pre_reset
        return out[0] if squeeze else out

    def clear_cache(self) -> None:
        """Drop the cached tensors (they may hold a live autograd graph).

        Needed before ``deepcopy`` of the network (the semantic memories copy it):
        a non-leaf tensor cannot be deep-copied.
        """
        last_v = getattr(self, "last_v", None)
        if isinstance(last_v, torch.Tensor):
            self.last_v = last_v.detach()
        self.last_spike = None


class RateMonitor:
    """Collects per-layer average firing rates of the last forward (Eq. 7/12)."""

    def __init__(self) -> None:
        self.rates: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def attach(self, model: nn.Module) -> "RateMonitor":
        for name, module in model.named_modules():
            if isinstance(module, LIFNode):
                self._handles.append(module.register_forward_hook(self._hook(name)))
        model.register_forward_pre_hook(lambda *a: self.rates.clear())
        return self

    def _hook(self, name: str):
        def hook(_module, _inputs, out: torch.Tensor):
            if out.dim() == 5:                       # (T, B, C, H, W)
                self.rates[name] = out.mean(dim=0).mean(dim=(2, 3))
            elif out.dim() == 3:                     # (T, B, C)
                self.rates[name] = out.mean(dim=0)
        return hook

    def clear(self) -> None:
        self.rates.clear()

    def mean_rates(self) -> dict[str, torch.Tensor]:
        return {k: v.mean(dim=0) for k, v in self.rates.items()}
