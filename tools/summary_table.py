"""Print every finished / partial run in runs/ as one table."""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = "seq_cifar10_T4_w32_buf500_"


def tag_of(path: str) -> str:
    return os.path.basename(os.path.dirname(path)).replace(PREFIX, "")


def main() -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "results.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows.append((tag_of(path), d.get("cil_acc"), d.get("til_acc"),
                     d.get("forgetting_cil"), d.get("forgetting_til"),
                     (d.get("elapsed_s") or 0) / 60))
    print(f"{'run':<14}{'class-IL':>9}{'task-IL':>9}{'F_cil':>8}{'F_til':>7}{'min':>7}")
    print("-" * 54)
    for r in sorted(rows, key=lambda x: -x[1]):
        print(f"{r[0]:<14}{r[1]:9.2f}{r[2]:9.2f}{r[3]:8.2f}{r[4]:7.2f}{r[5]:7.1f}")

    print("\npartial runs:")
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "progress.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d["tasks_done"] >= 5:
            continue
        last = d["history"][-1]["eval"]
        print(f"  {tag_of(path):<18} stopped after {d['tasks_done']} task(s), "
              f"class-IL={last['cil_acc']:.2f} task-IL={last['til_acc']:.2f}")

    cmp_path = os.path.join(ROOT, "runs", f"{PREFIX}snn_e25", "head_compare.json")
    if os.path.exists(cmp_path):
        with open(cmp_path, encoding="utf-8") as f:
            cmp = json.load(f)
        print("\nhead comparison on the snn_e25 checkpoint (same weights):")
        for head, v in cmp.items():
            print(f"  {head:<12} class-IL={v['cil']:6.2f} task-IL={v['til']:6.2f} "
                  f"per-task={v['per_task_cil']}")


if __name__ == "__main__":
    main()
