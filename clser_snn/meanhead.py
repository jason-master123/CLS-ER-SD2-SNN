"""Class-mean (nearest-class-mean) head used **during training**, not only at eval.

Why this exists
---------------
Every result in this project was obtained with a learnable linear head, and §8i of
the report shows that head carries a class-incremental bias whose size swings by
more than 30 points across configurations. Re-scoring the same checkpoints with a
class-mean head moves the ranking around completely.

The obvious next question is whether *training* with the bias-free head helps too:
if the classifier is a set of class means in feature space, the network cannot
exploit "new class gets a bigger logit" shortcuts, and the gradient is forced to
shape the features themselves.

Design
------
* ``MeanHead`` holds one prototype per class plus a ``present`` mask, so classes
  that have not been seen yet contribute no logit (they are masked to a large
  negative value rather than contributing noise).
* Prototypes are rebuilt **exactly** from the replay buffer at task/epoch
  boundaries. Rebuilding (instead of EMA) matters: an EMA of prototypes in a
  rotating feature space cancels itself out (measured earlier in this project:
  the prototype loss got stuck at log 2).
* The head is parameter-free, so it is not part of the optimizer and cannot drift
  between refreshes.

The head is exposed as a plain callable so the same code scores the working model
and the two EMA memories (each with its own prototypes, since each has its own
feature space).

Cold start
----------
A pure class-mean head cannot learn a class it has no prototype for: the logit is
masked to −1e4 and the class receives no gradient. At the beginning of a task the
buffer only holds *old* classes, so the trainer rebuilds the prototypes after the
first two batches — by then new-class samples are in the buffer and every class has
a mean. Without that rebuild the first epoch of every task would be wasted.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG = -1e4


class MeanHead(nn.Module):
    """Non-parametric classifier: score = similarity(feature, class mean)."""

    def __init__(self, n_classes: int, dim: int, metric: str = "cosine",
                 temperature: float = 10.0) -> None:
        super().__init__()
        metric = str(metric).lower()
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"unknown metric {metric!r}")
        self.n_classes = int(n_classes)
        self.dim = int(dim)
        self.metric = metric
        self.temperature = float(temperature)
        self.register_buffer("prototypes", torch.zeros(self.n_classes, self.dim))
        self.register_buffer("present", torch.zeros(self.n_classes, dtype=torch.bool))

    # ------------------------------------------------------------------ fit
    @torch.no_grad()
    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> int:
        """Rebuild every prototype from ``(features, labels)``; returns #classes."""
        feats = features.detach()
        labels = labels.detach()
        if self.metric == "cosine":
            feats = F.normalize(feats, dim=-1)
        self.prototypes.zero_()
        self.present.zero_()
        n = 0
        for cls in torch.unique(labels).tolist():
            mask = labels == cls
            if int(mask.sum()) == 0:
                continue
            self.prototypes[cls] = feats[mask].mean(dim=0)
            self.present[cls] = True
            n += 1
        if self.metric == "cosine":
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))
        return n

    # ---------------------------------------------------------------- score
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., K) logits, with unseen classes masked out."""
        feat = features
        if self.metric == "cosine":
            feat = F.normalize(feat, dim=-1)
            sim = feat @ self.prototypes.t()
        else:
            # negative squared distance, so larger is better (like a logit)
            sim = -torch.cdist(feat.reshape(-1, feat.shape[-1]),
                               self.prototypes).pow(2)
            sim = sim.reshape(*feat.shape[:-1], self.n_classes)
        logits = sim * self.temperature
        mask = self.present.reshape(*([1] * (logits.dim() - 1)), self.n_classes)
        return logits.masked_fill(~mask, _NEG)

    # ------------------------------------------------------------- helpers
    @property
    def num_present(self) -> int:
        return int(self.present.sum())

    # ------------------------------------------------------- batch bootstrap
    @torch.no_grad()
    def _prototypes_with_batch(self, features: torch.Tensor, labels: torch.Tensor
                               ) -> tuple:
        """Stored prototypes, with missing classes filled from *this batch*.

        The replay buffer is a reservoir over the whole stream, so early in a task
        it holds almost no samples of the new classes: their prototypes are missing
        and their logits are masked to −1e4 — which turns the training loss into
        +1e4 instead of "no signal" (measured: an auxiliary CE of 7665). Filling
        those rows from the current batch (detached) gives every present class a
        usable prototype while stored classes keep their buffer-based means.
        """
        protos = self.prototypes.clone()
        present = self.present.clone()
        feats = features.detach()
        if self.metric == "cosine":
            feats = F.normalize(feats, dim=-1)
        for cls in torch.unique(labels).tolist():
            if present[cls]:
                continue
            mask = labels == cls
            if int(mask.sum()) == 0:
                continue
            protos[cls] = feats[mask].mean(dim=0)
            present[cls] = True
        if self.metric == "cosine":
            protos = F.normalize(protos, dim=-1)
        return protos, present

    def logits_with_batch(self, features: torch.Tensor, labels: torch.Tensor
                          ) -> torch.Tensor:
        """Like ``forward`` but tolerates classes the buffer has not covered yet.

        ``features`` may be ``(D,)``/``(B, D)`` or a per-timestep ``(T, B, D)``
        stack; in the latter case ``labels`` (B,) is tiled over the time axis.
        """
        if features.dim() > 2:
            T = features.shape[0]
            flat = features.reshape(-1, features.shape[-1])
            labels_flat = labels.repeat(T)
            out = self.logits_with_batch(flat, labels_flat)
            return out.reshape(*features.shape[:-1], self.n_classes)
        protos, present = self._prototypes_with_batch(features, labels)
        feat = features
        if self.metric == "cosine":
            feat = F.normalize(feat, dim=-1)
            sim = feat @ protos.t()
        else:
            sim = -torch.cdist(feat.reshape(-1, feat.shape[-1]), protos).pow(2)
            sim = sim.reshape(*feat.shape[:-1], self.n_classes)
        logits = sim * self.temperature
        mask = present.reshape(*([1] * (logits.dim() - 1)), self.n_classes)
        return logits.masked_fill(~mask, _NEG)

    def summary(self) -> str:
        if self.num_present == 0:
            return "empty"
        norms = self.prototypes[self.present].norm(dim=-1)
        return (f"{self.num_present} prototypes, {self.metric}, "
                f"|p|∈[{norms.min():.2f},{norms.max():.2f}], T={self.temperature:g}")


class HeadBank(nn.Module):
    """One ``MeanHead`` per model copy (working / stable / plastic)."""

    def __init__(self, n_classes: int, dim: int, models=("working", "stable", "plastic"),
                 metric: str = "cosine", temperature: float = 10.0) -> None:
        super().__init__()
        self.names = tuple(models)
        for name in self.names:
            self.add_module(name, MeanHead(n_classes, dim, metric, temperature))

    def __getitem__(self, name: str) -> MeanHead:
        return getattr(self, name)

    def items(self):
        return ((name, getattr(self, name)) for name in self.names)


@torch.no_grad()
def fit_heads(bank: HeadBank, model, buffer, encode_fn, names: Optional[tuple] = None,
              batch_size: int = 64) -> dict:
    """Rebuild the prototypes of the requested model copies from the buffer.

    ``model`` is the working network; the memories live in ``model.memories`` (an
    object with ``stable`` / ``plastic`` attributes). Everything runs under
    ``no_grad`` — prototypes are targets, never differentiated through.
    """
    if buffer is None or len(buffer) == 0:
        return {}
    examples, labels = buffer.get_all_data()
    sources = {"working": model.net,
               "stable": getattr(getattr(model, "memories", None), "stable", None),
               "plastic": getattr(getattr(model, "memories", None), "plastic", None)}
    info = {}
    for name in (names or bank.names):
        net = sources.get(name)
        if net is None:
            continue
        feats = []
        for start in range(0, examples.shape[0], batch_size):
            chunk = examples[start:start + batch_size]
            spk = encode_fn(chunk)
            net.eval()
            feats.append(net.features(spk).mean(dim=0))
        pooled = torch.cat(feats, dim=0) if feats else None
        if pooled is None:
            continue
        info[name] = bank[name].fit(pooled.to(labels.device), labels)
    return info
