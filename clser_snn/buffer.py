"""Episodic memory: reservoir-sampled replay buffer.

A faithful port of ``utils/buffer.py`` from the official CLS-ER implementation
(``reservoir`` sampling, ``num_seen_examples`` bookkeeping), specialised to what
this project needs: un-augmented images + labels, stored on the device.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """Return the slot to overwrite for the current example, or -1 to skip it."""
    if num_seen_examples < buffer_size:
        return num_seen_examples
    rand = np.random.randint(0, num_seen_examples + 1)
    return rand if rand < buffer_size else -1


class ReservoirBuffer:
    def __init__(self, buffer_size: int = 500, device: str = "cuda") -> None:
        self.buffer_size = int(buffer_size)
        self.device = device
        self.num_seen_examples = 0
        self.examples: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self.num_seen_examples == 0 else \
            min(self.num_seen_examples, self.buffer_size)

    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    def _init(self, examples: torch.Tensor) -> None:
        self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]),
                                    dtype=examples.dtype, device=self.device)
        self.labels = torch.zeros(self.buffer_size, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def add_data(self, examples: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of un-augmented samples (``not_aug_inputs`` in CLS-ER)."""
        if self.examples is None:
            self._init(examples)
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                self.labels[index] = labels[i].to(self.device)

    def get_data(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample ``size`` distinct exemplars."""
        n = len(self)
        if n == 0:
            raise RuntimeError("buffer is empty")
        size = min(size, n)
        choice = np.random.choice(n, size=size, replace=False)
        choice = torch.from_numpy(choice).to(self.device)
        return self.examples[choice], self.labels[choice]

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(self)
        return self.examples[:n], self.labels[:n]

    def state_dict(self) -> dict:
        return {"buffer_size": self.buffer_size,
                "num_seen_examples": self.num_seen_examples,
                "examples": None if self.examples is None else self.examples.clone(),
                "labels": None if self.labels is None else self.labels.clone()}

    def load_state_dict(self, sd: dict) -> None:
        self.buffer_size = sd["buffer_size"]
        self.num_seen_examples = sd["num_seen_examples"]
        self.examples = None if sd["examples"] is None else sd["examples"].clone().to(self.device)
        self.labels = None if sd["labels"] is None else sd["labels"].clone().to(self.device)
