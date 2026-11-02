"""Class prototypes in feature space, used both as a training objective and as a
ready-made classifier.

Why this exists: the NCM experiment (README §9) showed that swapping the biased
linear head for prototypes recovers only +2.1 points, while task-IL stays at 89.
That reframes the remaining gap — the features *do* separate classes within a
task, but they do not place the tasks consistently in one space. Post-hoc fixes
cannot repair that; the constraint has to be part of training.

So: keep a prototype per class in the *same* feature space the linear head
consumes, and pull every sample (current **and** replayed) towards its own
prototype with a cosine softmax loss. Replayed exemplars of old classes are
therefore constantly re-anchored to their prototypes, which is exactly the
cross-task structure that was missing.

**Why the prototypes are recomputed instead of EMA-updated.** The first version
kept ``proto_c ← m·proto_c + (1−m)·normalize(class_mean)`` per step (m = 0.9). That
is wrong here: early in training the feature space itself rotates quickly, and an
EMA of *normalised* directions taken from slightly different feature spaces
cancels out — the running prototype drifts towards a near-random direction. The
symptom was unmistakable (README §10): the prototype loss sat at **0.70 ≈ log 2**
for a two-class task, i.e. the features were orthogonal to their own prototypes,
while the same term was destroying the classifier (CE 0.27 → 0.88).

So the prototypes are now rebuilt exactly from the episodic buffer at every epoch
boundary with the *current* model — one 500-sample forward pass, i.e. about 1.6
batches per epoch. This also makes the training targets identical to what
``--eval-head ncm`` uses at evaluation time.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class PrototypeMemory:
    def __init__(self, n_classes: int, dim: int, momentum: float = 0.9,
                 temperature: float = 0.1, device: str = "cpu") -> None:
        self.n_classes = int(n_classes)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.prototype = torch.zeros(self.n_classes, self.dim, device=device)
        self.present = torch.zeros(self.n_classes, dtype=torch.bool, device=device)

    # --------------------------------------------------------------- update
    @torch.no_grad()
    def refresh(self, model, examples: torch.Tensor, labels: torch.Tensor,
                encode_fn=None, min_samples: int = 4) -> int:
        """Rebuild the prototypes exactly from the buffer, with the current model.

        Returns the number of classes with a prototype. ``min_samples`` guards
        against a one-sample mean becoming a target.
        """
        if examples.numel() == 0:
            return self.n_present
        device = model.device if hasattr(model, "device") else "cpu"
        examples = examples.to(device)
        labels = labels.to(device)
        feats = model.net.features(
            encode_fn(examples) if encode_fn is not None else examples)
        feats = F.normalize(feats.mean(dim=0).float(), dim=1)      # (N, C)

        for c in labels.unique().tolist():
            if c < 0 or c >= self.n_classes:
                continue
            mask = labels == c
            if int(mask.sum()) < min_samples:
                continue
            mean = feats[mask].mean(dim=0)
            self.prototype[c] = F.normalize(mean, dim=0)
            self.present[c] = True
        return self.n_present

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA of the per-class mean of the **detached**, L2-normalised features.

        Kept for reference / ablations only: on a rotating feature space this
        average cancels out (see the module docstring), so training uses
        :meth:`refresh` instead.
        """
        f = F.normalize(feats.detach().float(), dim=1)
        labels = labels.to(f.device)
        for c in labels.unique().tolist():
            if c < 0 or c >= self.n_classes:
                continue
            mean = f[labels == c].mean(dim=0)
            mean = F.normalize(mean, dim=0)
            if self.present[c]:
                self.prototype[c].mul_(self.momentum).add_(mean, alpha=1 - self.momentum)
                self.prototype[c] = F.normalize(self.prototype[c], dim=0)
            else:
                self.prototype[c] = mean
                self.present[c] = True

    # -------------------------------------------------------------- scoring
    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Cosine similarity scaled by 1/temperature; absent classes score -inf."""
        f = F.normalize(feats.float(), dim=1)
        sim = f @ F.normalize(self.prototype, dim=1).t()
        sim = sim / max(self.temperature, 1e-6)
        absent = ~self.present
        if bool(absent.any()):
            sim = sim.masked_fill(absent.unsqueeze(0), float("-inf"))
        return sim

    def has(self, labels: torch.Tensor) -> torch.Tensor:
        """Mask of samples whose class already has a prototype."""
        labels = labels.to(self.present.device).long()
        valid = torch.zeros_like(labels, dtype=torch.bool)
        in_range = (labels >= 0) & (labels < self.n_classes)
        valid[in_range] = self.present[labels[in_range]]
        return valid

    @property
    def n_present(self) -> int:
        return int(self.present.sum())

    # ----------------------------------------------------------------- misc
    def to(self, device) -> "PrototypeMemory":
        self.prototype = self.prototype.to(device)
        self.present = self.present.to(device)
        return self

    def state_dict(self) -> dict:
        return {"prototype": self.prototype.detach().cpu(),
                "present": self.present.detach().cpu(),
                "momentum": self.momentum, "temperature": self.temperature}

    @classmethod
    def from_state(cls, sd: dict) -> "PrototypeMemory":
        mem = cls(sd["prototype"].shape[0], sd["prototype"].shape[1],
                  momentum=sd.get("momentum", 0.9),
                  temperature=sd.get("temperature", 0.1))
        mem.prototype = sd["prototype"].clone()
        mem.present = sd["present"].clone()
        return mem

    def summary(self) -> str:
        if self.n_present == 0:
            return "no prototypes"
        return (f"{self.n_present}/{self.n_classes} prototypes, "
                f"momentum={self.momentum}, τ={self.temperature}")

    # ------------------------------------------------------------- loss util
    def loss(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cosine-softmax CE against the prototypes, on samples that have one.

        Classes without a prototype score ``-inf`` everywhere, so they are dropped
        from the softmax (equivalently: their columns are ignored). Samples whose
        own label has no prototype yet are skipped entirely.
        """
        valid = self.has(labels)
        if self.n_present < 2 or not bool(valid.any()):
            return torch.zeros((), device=feats.device)
        logits = self.logits(feats)[valid]
        return F.cross_entropy(logits, labels.to(feats.device)[valid])

    def margin_loss(self, feats: torch.Tensor, labels: torch.Tensor,
                    margin: float = 0.1) -> torch.Tensor:
        """Inter-class hinge on the cosine gap (the missing "negative" term).

        For every sample, require ``cos(f, proto_y) - max_{j≠y} cos(f, proto_j) ≥ m``.
        The attraction term above only tightens each cluster; it never says anything
        about *where* the clusters sit relative to one another, which is precisely
        what class-incremental learning breaks. This term does.

        Note that a prototype-vs-prototype penalty would be constant inside an epoch
        (prototypes only change on ``refresh``) and therefore carry no gradient, so
        the hinge is written on the *features* instead.
        """
        valid = self.has(labels)
        labels = labels.to(feats.device)
        if self.n_present < 2 or not bool(valid.any()):
            return torch.zeros((), device=feats.device)

        f = F.normalize(feats.float(), dim=1)
        sim = f @ F.normalize(self.prototype, dim=1).t()          # (B, K)
        sim = sim.masked_fill(~self.present.unsqueeze(0), float("-inf"))

        rows = torch.arange(sim.shape[0], device=sim.device)
        own = sim[rows, labels.clamp(0, self.n_classes - 1)]
        other = sim.clone()
        other[rows, labels.clamp(0, self.n_classes - 1)] = float("-inf")
        hardest = other.max(dim=1).values
        finite = torch.isfinite(hardest) & torch.isfinite(own) & valid
        if not bool(finite.any()):
            return torch.zeros((), device=feats.device)
        hinge = F.relu(margin - (own - hardest)[finite])
        return hinge.mean()
