#!/usr/bin/env python
"""Train the CLS-ER + SD^2-SNN hybrid on a continual-learning stream.

Examples
--------
CLS-ER scheme on the spiking substrate, Seq-CIFAR-10 (5 x 2 classes, buffer 500,
the setting where the CLS-ER paper reports 75.22 % class-IL with a ReLU ResNet-18)::

    python scripts/train.py --dataset seq_cifar10 --epochs 50 --buffer-size 500

Same, with the SD^2-SNN regularisers layered on::

    python scripts/train.py --dataset seq_cifar10 --decompose --rank-ratio 0.75 \
        --stability-weight 1e-3 --sparse-weight 1e-3 --subspace-weight 1.0

Seq-CIFAR-100 (10 x 10 classes)::

    python scripts/train.py --dataset seq_cifar100 --epochs 50

Quick smoke run on a subsampled stream (a few minutes on GPU)::

    python scripts/train.py --dataset seq_cifar10 --n-tasks 3 --epochs 1 \
        --limit-train 300 --limit-test 100 --buffer-size 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clser_snn.config import HybridConfig, add_config_args
from clser_snn.datasets import build_tasks
from clser_snn.model import SpikingCLSER
from clser_snn.trainer import Trainer


def parse() -> HybridConfig:
    p = argparse.ArgumentParser(description="CLS-ER + SD^2-SNN")
    p = add_config_args(p)
    p.add_argument("--tag", type=str, default="", help="suffix for the run directory")
    args = p.parse_args()

    cfg = HybridConfig()
    cfg.tag = ""                       # not a dataclass field: run-name suffix only
    for key, value in vars(args).items():
        if key == "tag" or hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def default_name(cfg: HybridConfig) -> str:
    parts = [cfg.dataset, f"T{cfg.time_steps}", f"w{cfg.width}", f"buf{cfg.buffer_size}"]
    if cfg.reg_weight != 0.75:
        parts.append(f"reg{cfg.reg_weight:g}")
    if cfg.decompose:
        parts.append("dec")
    if cfg.sparse_weight:
        parts.append("sp")
    if cfg.subspace_weight:
        parts.append("sub")
    if cfg.eval_mode != "working":
        parts.append(cfg.eval_mode)
    return "_".join(parts)


def main() -> int:
    cfg = parse()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        cfg.device = "cpu"
    cfg = cfg.resolved()
    if not cfg.run_name:
        cfg.run_name = default_name(cfg)
    if cfg.tag:
        cfg.run_name = f"{cfg.run_name}_{cfg.tag}"
    os.makedirs(os.path.join(cfg.output_dir, cfg.run_name), exist_ok=True)

    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg)
    # the classifier must cover every class of the stream (splits may be uneven)
    cfg.num_classes = max(c for task in tasks for c in task.classes) + 1
    cfg.classes_per_task = len(tasks[0].classes)
    model = SpikingCLSER(cfg)
    trainer = Trainer(cfg, model, tasks)
    final = trainer.fit()

    print("\n=== summary ===")
    for key in ("cil_acc", "til_acc", "forgetting_cil", "forgetting_til", "bwt"):
        print(f"{key:>16}: {final[key]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
