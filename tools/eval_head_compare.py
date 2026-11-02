"""Re-evaluate a trained checkpoint under different inference heads.

Cheap experiment (a few minutes, no training): the same weights and the same
episodic buffer are scored with the linear head and with the NCM prototypes, so
the only variable is the *decision rule*.

Usage:
    python tools/eval_head_compare.py runs/seq_cifar10_T4_w32_buf500_snn_e25
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clser_snn.config import HybridConfig
from clser_snn.datasets import build_tasks, task_loader
from clser_snn.metrics import evaluate_task, summarise
from clser_snn.model import SpikingCLSER, encode_direct
from clser_snn.ncm import NCMClassifier


def load_run(run_dir: str) -> HybridConfig:
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu",
                    weights_only=False)
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    cfg = HybridConfig(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in fields})
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.save_ckpt = False
    cfg.num_workers = 0
    return cfg, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ncm-temperature", type=float, default=10.0)
    args = ap.parse_args()

    cfg, ck = load_run(args.run_dir)
    tasks = build_tasks(cfg)
    model = SpikingCLSER(cfg)
    model.load_state_dict(ck["model"])
    model.train_step_mode(False)
    print(f"[ckpt] {args.run_dir}  tasks_done={ck.get('task', -1) + 1}  "
          f"buffer={len(model.buffer)}  params={model.num_params / 1e6:.2f}M")

    seen = [c for t in tasks for c in t.classes]
    examples, labels = model.buffer.get_all_data()
    n_classes = int(model.net.fc.out_features)
    prototypes = {
        "ncm-cosine": NCMClassifier.fit(
            model, examples, labels, n_classes,
            encode_fn=lambda x: encode_direct(x, cfg.time_steps),
            temperature=args.ncm_temperature, metric="cosine"),
        "ncm-euclid": NCMClassifier.fit(
            model, examples, labels, n_classes,
            encode_fn=lambda x: encode_direct(x, cfg.time_steps),
            temperature=args.ncm_temperature, metric="euclidean"),
    }
    for name, ncm in prototypes.items():
        print(f"[{name}] {ncm.summary()}")

    results = {}
    for head in ["linear", "ncm-cosine", "ncm-euclid"]:
        ncm = prototypes.get(head)
        cil, til, per = [], [], []
        for i, task in enumerate(tasks):
            loader = task_loader(task, cfg, train=False)
            res = evaluate_task(model, loader, seen, task.classes, str(model.device),
                                cfg.time_steps, encoding="direct", passes=1, ncm=ncm)
            cil.append(res["cil"])
            til.append(res["til"])
        summary = summarise([cil], [til])
        results[head] = {"cil": sum(cil) / len(cil), "til": sum(til) / len(til),
                         "per_task_cil": [round(v, 1) for v in cil]}
        print(f"[{head:>10}] class-IL = {results[head]['cil']:6.2f}   "
              f"task-IL = {results[head]['til']:6.2f}   per-task {results[head]['per_task_cil']}")

    base = results["linear"]["cil"]
    print("\n=== delta vs linear head ===")
    for head, value in results.items():
        print(f"{head:>10}: {value['cil'] - base:+6.2f} class-IL")

    out = os.path.join(args.run_dir, "head_compare.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
