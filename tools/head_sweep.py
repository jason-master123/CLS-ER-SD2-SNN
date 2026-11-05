"""Re-score every finished 5-task run with the NCM heads, in one process.

The linear head has a well-known class-incremental bias (new classes get higher
logits), and the size of that bias depends strongly on the configuration: when we
re-scored four checkpoints by hand the gain ranged from +1.0 to +12.5 points. That
is large enough to reorder configurations, so every run is re-scored here and the
two rankings (linear vs NCM) are printed side by side.

Usage:
    python tools/head_sweep.py            # all runs with 5 tasks
    python tools/head_sweep.py --top 8    # only the best N by linear class-IL
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clser_snn.config import HybridConfig
from clser_snn.datasets import build_tasks, task_loader
from clser_snn.metrics import evaluate_task
from clser_snn.model import SpikingCLSER, encode_direct
from clser_snn.ncm import NCMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def finished_runs() -> list[tuple[str, dict]]:
    out = []
    for path in glob.glob(os.path.join(ROOT, "runs", "*", "results.json")):
        run_dir = os.path.dirname(path)
        with open(path) as f:
            res = json.load(f)
        if len(res.get("per_task_cil", [])) != 5:
            continue
        if not os.path.exists(os.path.join(run_dir, "last.pt")):
            continue
        out.append((run_dir, res))
    out.sort(key=lambda kv: kv[1]["cil_acc"], reverse=True)
    return out


def score(run_dir: str, tasks, cache: dict) -> dict:
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu",
                    weights_only=False)
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    cfg = HybridConfig(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in fields})
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.save_ckpt = False
    cfg.num_workers = 0

    model = SpikingCLSER(cfg)
    model.load_state_dict(ck["model"])
    model.train_step_mode(False)

    # task splits depend only on the dataset, so build them once
    key = (cfg.dataset, cfg.n_tasks, cfg.classes_per_task)
    if key not in cache:
        cache[key] = build_tasks(cfg)
    tasks_here = cache[key]

    seen = [c for t in tasks_here for c in t.classes]
    examples, labels = model.buffer.get_all_data()
    n_classes = int(model.net.fc.out_features)
    ncm_cos = NCMClassifier.fit(
        model, examples, labels, n_classes,
        encode_fn=lambda x: encode_direct(x, cfg.time_steps),
        temperature=10.0, metric="cosine")
    ncm_euc = NCMClassifier.fit(
        model, examples, labels, n_classes,
        encode_fn=lambda x: encode_direct(x, cfg.time_steps),
        temperature=10.0, metric="euclidean")

    out = {}
    for name, ncm in (("linear", None), ("ncm-cosine", ncm_cos), ("ncm-euclid", ncm_euc)):
        cil, til = [], []
        for task in tasks_here:
            loader = task_loader(task, cfg, train=False)
            res = evaluate_task(model, loader, seen, task.classes, str(model.device),
                                cfg.time_steps, encoding="direct", passes=1, ncm=ncm)
            cil.append(res["cil"])
            til.append(res["til"])
        out[name] = {"cil": sum(cil) / len(cil), "til": sum(til) / len(til),
                     "per_task_cil": [round(v, 1) for v in cil]}
    with open(os.path.join(run_dir, "head_compare.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="only the best N runs (0 = all)")
    args = ap.parse_args()

    runs = finished_runs()
    if args.top:
        runs = runs[: args.top]

    rows, cache = [], {}
    for run_dir, res in runs:
        name = os.path.basename(run_dir)
        try:
            scored = score(run_dir, None, cache)
        except Exception as exc:                       # keep going on a bad checkpoint
            print(f"[skip] {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append((name, res, scored))
        best = max(scored.values(), key=lambda d: d["cil"])
        print(f"{name[-42:]:>42}  linear {res['cil_acc']:5.2f}  "
              f"cos {scored['ncm-cosine']['cil']:5.2f}  "
              f"euc {scored['ncm-euclid']['cil']:5.2f}   "
              f"best {best['cil']:5.2f} ({best['cil'] - res['cil_acc']:+.2f})",
              flush=True)

    print("\n=== ranking by best-NCM class-IL ===")
    ranked = sorted(rows, key=lambda r: max(v["cil"] for v in r[2].values()), reverse=True)
    print(f"{'config':>44} {'linear':>7} {'NCM':>7} {'gain':>7}")
    for name, res, scored in ranked:
        best = max(scored.values(), key=lambda d: d["cil"])
        print(f"{name[-44:]:>44} {res['cil_acc']:7.2f} {best['cil']:7.2f} "
              f"{best['cil'] - res['cil_acc']:+7.2f}")

    with open(os.path.join(ROOT, "report", "head_sweep.json"), "w") as f:
        json.dump([{"run": n, "linear": r["cil_acc"],
                    "ncm_cosine": s["ncm-cosine"]["cil"],
                    "ncm_euclid": s["ncm-euclid"]["cil"],
                    "per_task": {"linear": s["linear"]["per_task_cil"],
                                 "ncm_cosine": s["ncm-cosine"]["per_task_cil"],
                                 "ncm_euclid": s["ncm-euclid"]["per_task_cil"]}}
                   for n, r, s in rows], f, indent=2)
    print(f"\nsaved {os.path.join(ROOT, 'report', 'head_sweep.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
