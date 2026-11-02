"""Collect every run of the CLS-ER + SD^2-SNN project into report/results.json."""

from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = re.compile(r"^seq_cifar10_(T\d+_)?w\d+_buf\d+_")


def tag_of(path: str) -> str:
    name = os.path.basename(os.path.dirname(path))
    return PREFIX.sub("", name)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "progress.json"))):
        tag = tag_of(path)
        prog = load(path)
        results_path = os.path.join(os.path.dirname(path), "results.json")
        final = load(results_path) if os.path.exists(results_path) else {}
        runs[tag] = {
            "tasks_done": prog["tasks_done"],
            "cil_matrix": prog["cil_matrix"],
            "til_matrix": prog["til_matrix"],
            "history": [
                {"task": h["task"], "classes": h["classes"],
                 "eval": h["eval"], "train": {k: h["train"].get(k) for k in
                                              ("loss", "ce", "cons", "cons_w", "proto", "margin")},
                 "post": h.get("post")}
                for h in prog["history"]
            ],
            "final": {k: final.get(k) for k in
                      ("cil_acc", "til_acc", "forgetting_cil", "forgetting_til", "bwt",
                       "elapsed_s", "per_task_cil", "per_task_til")} if final else None,
        }

    extra = {}
    cmp_path = os.path.join(ROOT, "runs", "seq_cifar10_T4_w32_buf500_snn_e25",
                            "head_compare.json")
    if os.path.exists(cmp_path):
        extra["head_compare"] = load(cmp_path)

    out = os.path.join(ROOT, "report", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "extra": extra}, f, indent=2, ensure_ascii=False)

    print(f"{len(runs)} runs -> {out}")
    for tag, r in sorted(runs.items(), key=lambda kv: -(kv[1]["final"] or {}).get("cil_acc", 0)):
        f = r["final"]
        line = (f"  {tag:<22} {r['tasks_done']} tasks  " +
                (f"{f['cil_acc']:6.2f} / {f['til_acc']:6.2f}" if f
                 else f"{r['history'][-1]['eval']['cil_acc']:6.2f} / "
                      f"{r['history'][-1]['eval']['til_acc']:6.2f}  (partial)"))
        print(line)


if __name__ == "__main__":
    main()
