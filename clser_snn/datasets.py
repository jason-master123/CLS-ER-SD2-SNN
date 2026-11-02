"""Continual task streams: Seq-CIFAR-10 and Seq-CIFAR-100.

Each item is returned as a **triple** ``(x_aug, y, x_raw)``, exactly like the
Mammoth-based CLS-ER implementation: the episodic buffer stores ``x_raw`` (the
normalised but un-augmented image) and re-augments it every time it is replayed.

Stream presets match the benchmarks used in the CLS-ER paper:

=============  ========  =================
preset         tasks     classes per task
=============  ========  =================
seq_cifar10    5         2
seq_cifar100   10        10
seq_cifar100_20 20       5
=============  ========  =================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

STREAM_PRESETS = {
    "seq_cifar10": {"n_tasks": 5, "classes_per_task": 2, "num_classes": 10},
    "seq_cifar100": {"n_tasks": 10, "classes_per_task": 10, "num_classes": 100},
    "seq_cifar100_20": {"n_tasks": 20, "classes_per_task": 5, "num_classes": 100},
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def base_transform() -> Callable:
    """Normalisation only: this is what the episodic buffer stores."""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def aug_transform() -> Callable:
    """Augmentation applied to tensors (CIFAR convention: crop + flip)."""
    return transforms.Compose([transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip()])


@dataclass
class Task:
    task_id: int
    classes: List[int]
    train: Dataset
    test: Dataset


class _TripleDataset(Dataset):
    """Wraps a torchvision dataset and returns (x_aug, y, x_raw)."""

    def __init__(self, base: Dataset, indices: Sequence[int],
                 base_tf: Callable, aug_tf: Optional[Callable]) -> None:
        self.base = base
        self.indices = list(indices)
        self.base_tf = base_tf
        self.aug_tf = aug_tf

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, y = self.base[self.indices[i]]           # PIL image from torchvision
        raw = self.base_tf(img)
        x = self.aug_tf(raw) if self.aug_tf is not None else raw
        return x, int(y), raw


def _labels_of(base: Dataset) -> np.ndarray:
    if hasattr(base, "targets"):
        return np.asarray(base.targets)
    if hasattr(base, "labels"):
        return np.asarray(base.labels)
    raise AttributeError("cannot read labels")


def _cifar10_root(data_root: str) -> str:
    candidates = [os.path.join(data_root, "CIFAR10"),
                  os.path.join(data_root, "cifar-10-batches-py"),
                  os.path.join("..", "CLS-ER", "datasets", "CIFAR10"),
                  os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                               "CLS-ER", "datasets", "CIFAR10"))]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "cifar-10-batches-py")):
            return c
    return candidates[0]


def _cifar100_root(data_root: str) -> str:
    candidates = [os.path.join(data_root, "CIFAR100"),
                  os.path.join(data_root, "cifar-100-python"),
                  os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                               "SD2-SNN-repro", "data", "c100root")),
                  os.path.join("..", "SD2-SNN-repro", "data", "c100root"),
                  os.path.join(data_root, "cifar-100-batches")]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "cifar-100-python")):
            return c
    return candidates[0]


def _load_base(cfg) -> Tuple[Dataset, Dataset]:
    if cfg.dataset.startswith("seq_cifar10"):
        from torchvision.datasets import CIFAR10
        root = _cifar10_root(cfg.data_root)
        os.makedirs(root, exist_ok=True)
        return (CIFAR10(root, True, download=True),
                CIFAR10(root, False, download=True))
    if cfg.dataset.startswith("seq_cifar100"):
        from torchvision.datasets import CIFAR100
        root = _cifar100_root(cfg.data_root)
        os.makedirs(root, exist_ok=True)
        return (CIFAR100(root, True, download=True),
                CIFAR100(root, False, download=True))
    raise ValueError(f"unknown dataset {cfg.dataset}")


def class_splits(classes: Sequence[int], n_tasks: int) -> List[List[int]]:
    classes = list(classes)
    per = max(1, len(classes) // n_tasks)
    out, start = [], 0
    for k in range(n_tasks):
        take = per if k < n_tasks - 1 else len(classes) - start
        out.append(classes[start:start + take])
        start += take
    return out


def build_tasks(cfg) -> List[Task]:
    cfg = cfg.resolved()
    train_base, test_base = _load_base(cfg)
    num_classes = len(np.unique(_labels_of(train_base)))
    splits = class_splits(range(num_classes), cfg.n_tasks)

    base_tf = base_transform()
    train_aug = aug_transform()
    train_labels = _labels_of(train_base)
    test_labels = _labels_of(test_base)

    tasks: List[Task] = []
    for k, classes in enumerate(splits):
        tr_idx = np.nonzero(np.isin(train_labels, classes))[0]
        te_idx = np.nonzero(np.isin(test_labels, classes))[0]
        if cfg.limit_train:
            tr_idx = tr_idx[: cfg.limit_train]
        if cfg.limit_test:
            te_idx = te_idx[: cfg.limit_test]
        tasks.append(Task(
            task_id=k, classes=list(classes),
            train=_TripleDataset(train_base, tr_idx, base_tf, train_aug),
            test=_TripleDataset(test_base, te_idx, base_tf, None),
        ))
    return tasks


def task_loader(task: Task, cfg, train: bool) -> DataLoader:
    nw = 0 if os.name == "nt" else cfg.num_workers
    return DataLoader(
        task.train if train else task.test,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=nw,
        pin_memory=True,
        drop_last=train and len(task.train) > cfg.batch_size,
    )
