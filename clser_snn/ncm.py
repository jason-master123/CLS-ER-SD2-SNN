"""Nearest-Class-Mean classifier for the spiking features.

The calibration experiments (README §8) showed that the *linear head* is what
loses the class-incremental points: task-IL reaches 89 % while class-IL stalls at
57 %, i.e. the features separate the classes but the head cannot compare tasks.
Post-hoc logit corrections made it worse, three times over.

NCM attacks the problem from the other side: it throws the biased head away at
inference time and classifies with **class prototypes in feature space**. Cosine
similarity to a prototype has no notion of "which task the class came from", so
the cross-task bias simply cannot arise. This is the classifier iCaRL used, and it
works with the same episodic buffer we already keep.

Prototypes must be recomputed with the *current* feature extractor before every
evaluation (features drift as tasks are added), which is what ``Trainer`` does.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class NCMClassifier:
    def __init__(self, prototypes: torch.Tensor, present: torch.Tensor,
                 temperature: float = 10.0, metric: str = "cosine") -> None:
        self.prototypes = prototypes            # (K, C), already normalised if cosine
        self.present = present.bool()           # (K,)
        self.temperature = float(temperature)
        self.metric = metric

    # ------------------------------------------------------------------ fit
    @classmethod
    @torch.no_grad()
    def fit(cls, model, examples: torch.Tensor, labels: torch.Tensor,
            n_classes: int, encode_fn=None, temperature: float = 10.0,
            metric: str = "cosine") -> "NCMClassifier":
        """Average the features of every class present in the buffer."""
        device = model.device
        feats = model.net.features(
            encode_fn(examples.to(device)) if encode_fn is not None
            else examples.to(device)).mean(dim=0)                  # (N, C)
        feats = feats.float()
        labels = labels.to(device)

        prototypes = torch.zeros(n_classes, feats.shape[1], device=device)
        present = torch.zeros(n_classes, dtype=torch.bool, device=device)
        for c in labels.unique().tolist():
            mask = labels == c
            prototypes[c] = feats[mask].mean(dim=0)
            present[c] = True
        if metric == "cosine":
            prototypes = F.normalize(prototypes, dim=1)
        return cls(prototypes.cpu(), present.cpu(), temperature, metric)

    # -------------------------------------------------------------- scoring
    @torch.no_grad()
    def scores(self, feats: torch.Tensor) -> torch.Tensor:
        """Similarity of every sample to every class prototype -> (B, K).

        Computed on the *features'* device (the prototypes live on CPU so the
        classifier can be stored in a checkpoint, but evaluation runs on GPU).
        """
        f = feats.float()
        device = f.device
        prototypes = self.prototypes.to(device)
        present = self.present.to(device)
        if self.metric == "cosine":
            f = F.normalize(f, dim=1)
            sim = f @ F.normalize(prototypes, dim=1).t()
        else:                                   # negative squared euclidean distance
            sim = -torch.cdist(f, prototypes).pow(2)
        absent = ~present
        if bool(absent.any()):
            sim = sim.masked_fill(absent.unsqueeze(0), float("-inf"))
        return sim

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Scaled scores, so the existing softmax/argmax path can be reused."""
        return self.scores(feats) * self.temperature

    # ----------------------------------------------------------------- misc
    @property
    def n_present(self) -> int:
        return int(self.present.sum())

    def state_dict(self) -> dict:
        return {"prototypes": self.prototypes.clone(), "present": self.present.clone(),
                "temperature": self.temperature, "metric": self.metric}

    @classmethod
    def from_state(cls, sd: dict) -> "NCMClassifier":
        return cls(sd["prototypes"], sd["present"], sd["temperature"], sd["metric"])

    def summary(self) -> str:
        if self.n_present == 0:
            return "empty"
        norms = self.prototypes[self.present].norm(dim=1)
        return (f"{self.n_present} prototypes, |proto|∈[{norms.min():.2f},{norms.max():.2f}], "
                f"{self.metric}")
