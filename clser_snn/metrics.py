"""Class-IL / Task-IL metrics for a *stochastic* spiking network.

Two differences from the ANN evaluation protocol:

* the forward pass can be repeated with independent stochastic encodings and the
  softmax probabilities averaged (``passes``) — a single pass of an SNN with a
  small T is a high-variance estimate of the model's belief;
* the encoding mode is explicit, so evaluation can be performed with the same
  deterministic direct coding used in training, or with a rate code.

``class-il``  — argmax over all classes seen so far (the setting CLS-ER reports)
``task-il``   — argmax restricted to the task's label space (``mask_classes``)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .model import encode_input


@torch.no_grad()
def evaluate_task(model, loader, seen_classes: Sequence[int], task_classes: Sequence[int],
                  device: str, time_steps: int, encoding: str = "direct",
                  passes: int = 1, calibrator=None, ncm=None) -> Dict[str, float]:
    """Accuracy on one task.

    Three decision rules are supported and are mutually exclusive by intent:

    * linear head + optional ``calibrator`` (post-hoc logit transform);
    * ``ncm``: nearest-class-mean over the backbone features (bypasses the head).
    """
    model.train_step_mode(False)
    seen = torch.as_tensor(list(seen_classes), device=device)
    task_c = torch.as_tensor(list(task_classes), device=device)
    correct_cil = correct_til = total = 0
    passes = max(1, int(passes))

    for x_aug, y, x_raw in loader:
        x_raw = x_raw.to(device)
        y = y.to(device)

        probs = None
        for _ in range(passes):
            spk = encode_input(x_raw, time_steps, encoding)
            if ncm is not None:
                feats = model.net.features(spk).mean(dim=0)
                logits = ncm.logits(feats)
            else:
                logits = model.net(spk)
                if calibrator is not None:
                    logits = calibrator.transform(logits)
            if probs is None:
                probs = torch.zeros(x_raw.shape[0], logits.shape[1], device=device)
            probs += F.softmax(logits, dim=1)
        probs /= passes

        seen_probs = probs.index_select(1, seen)
        correct_cil += (seen[seen_probs.argmax(dim=1)] == y).sum().item()

        task_probs = probs.index_select(1, task_c)
        correct_til += (task_c[task_probs.argmax(dim=1)] == y).sum().item()

        total += y.numel()

    total = max(1, total)
    return {"cil": 100.0 * correct_cil / total, "til": 100.0 * correct_til / total, "n": total}


def summarise(cil_matrix: List[List[float]], til_matrix: List[List[float]]) -> Dict[str, object]:
    """Aggregate the (K x K) accuracy matrices."""
    K = len(cil_matrix)
    final_cil = cil_matrix[-1]
    final_til = til_matrix[-1]

    f_cil, f_til = [], []
    for i in range(K):
        best_c = max(cil_matrix[j][i] for j in range(i, K))
        best_t = max(til_matrix[j][i] for j in range(i, K))
        f_cil.append(best_c - final_cil[i])
        f_til.append(best_t - final_til[i])

    bwt, n_pairs = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            bwt += cil_matrix[j][i] - cil_matrix[i][i]
            n_pairs += 1

    return {
        "cil_acc": sum(final_cil) / K,
        "til_acc": sum(final_til) / K,
        "forgetting_cil": sum(f_cil) / K,
        "forgetting_til": sum(f_til) / K,
        "bwt": bwt / n_pairs if n_pairs else 0.0,
        "per_task_cil": [round(v, 2) for v in final_cil],
        "per_task_til": [round(v, 2) for v in final_til],
    }
