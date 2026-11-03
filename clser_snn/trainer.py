"""Task-wise training / evaluation loop.

Follows CLS-ER's schedule (``utils/training.py`` of the official repo): one
optimiser, a per-task cosine (or constant) learning-rate schedule, evaluation on
*every task seen so far* after each task. The only addition is the bookkeeping the
spiking substrate needs (input encoding lives in the model) and the optional
SD^2-SNN task-boundary step (sparse subspace + basis snapshot).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import torch

from .datasets import Task, task_loader
from .metrics import evaluate_task, summarise


class Trainer:
    def __init__(self, cfg, model, tasks: List[Task], resume: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.tasks = tasks
        self.device = str(model.device)

        self.opt = model.opt                      # built once, in the model
        model.set_optimizer(self.opt)
        # remember the per-group base learning rates: the cosine schedule decays
        # them to ~0 within a task, and a fresh scheduler would otherwise adopt
        # those decayed values as its base (silently freezing later tasks).
        self.base_lrs = [group["lr"] for group in self.opt.param_groups]

        self.cil_matrix: List[List[float]] = []
        self.til_matrix: List[List[float]] = []
        self.history: List[dict] = []
        self.seen_classes: List[int] = []
        self.start_task = 0
        self.elapsed_offset = 0.0
        self._t0: Optional[float] = None

        # ---- resume from a checkpoint written after a completed task --------
        # ``last.pt`` holds the model (net + both memories + buffer + calibrator +
        # prototypes) and the accuracy matrices; ``progress.json`` holds the history.
        # Resuming therefore never re-trains a finished task.
        if resume:
            self.model.load_state_dict(resume["model"])
            self.cil_matrix = [list(r) for r in resume.get("cil", [])]
            self.til_matrix = [list(r) for r in resume.get("til", [])]
            self.history = list(resume.get("history", []))
            self.start_task = int(resume.get("task", -1)) + 1
            self.seen_classes = [c for t in self.tasks[:self.start_task] for c in t.classes]
            self.elapsed_offset = float(resume.get("elapsed_s", 0.0))
            print(f"[resume] restored after task {self.start_task - 1}; "
                  f"continuing at task {self.start_task}/{len(self.tasks)} "
                  f"(seen classes {self.seen_classes})", flush=True)

        self.run_dir = os.path.join(cfg.output_dir, cfg.run_name) if cfg.run_name else cfg.output_dir
        os.makedirs(self.run_dir, exist_ok=True)
        # dump the *exact* configuration next to the results, so any number in the
        # report can be traced back to the command that produced it
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(vars(cfg) if hasattr(cfg, "__dict__") else
                      {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")},
                      f, indent=2, default=str)

    # ------------------------------------------------------------- training
    def _reset_lr(self) -> None:
        """Restore the base learning rate at the start of every task."""
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base

    def train_one_task(self, k: int) -> Dict[str, float]:
        cfg = self.cfg
        loader = task_loader(self.tasks[k], cfg, train=True)
        model = self.model
        model.train_step_mode(True)

        self._reset_lr()
        steps = max(1, len(loader) * cfg.epochs)
        if cfg.lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=steps)
        else:
            sched = None
        if k == 0:
            print(f"[lr] base learning rates: "
                  f"{[round(v, 6) for v in self.base_lrs]}", flush=True)

        # re-augment replayed exemplars exactly like the stream samples
        from .datasets import aug_transform
        augment_fn = aug_transform()

        # prototypes must live in the *current* feature space: rebuild them before
        # the task starts and at every epoch boundary (cheap: ~1.6 batches)
        if getattr(model, "prototypes", None) is not None:
            info = model.refresh_prototypes()
            if info:
                print(f"[proto] start of task {k}: {info}", flush=True)

        last: Dict[str, float] = {}
        for epoch in range(cfg.epochs):
            agg: Dict[str, list] = {}
            n_batches = 0
            for x_aug, y, x_raw in loader:
                x_aug = x_aug.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                x_raw = x_raw.to(self.device, non_blocking=True)

                stats = model.observe(x_aug, y, x_raw, augment_fn)
                if sched is not None:
                    sched.step()
                for key, value in stats.items():
                    if value != value:          # skip NaN (e.g. no exemplars yet)
                        continue
                    bucket = agg.setdefault(key, [0.0, 0])
                    bucket[0] += value
                    bucket[1] += 1
                n_batches += 1

            last = {}
            for key, (total, count) in agg.items():
                last[key] = total / max(1, count)
            last["epoch"] = epoch + 1
            if getattr(model, "prototypes", None) is not None:
                model.refresh_prototypes()          # targets follow the features
            if (epoch + 1) % max(1, cfg.log_every) == 0 or epoch == 0:
                print(f"  [task {k} epoch {epoch + 1}/{cfg.epochs}] "
                      f"loss={last['loss']:.4f} ce={last['ce']:.4f} cons={last['cons']:.4f} "
                      f"proto={last.get('proto', float('nan')):.3f} "
                      f"frac_stable={last.get('frac_stable', float('nan')):.3f} "
                      f"buf={len(model.buffer)}", flush=True)
        return last

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_all(self, k: int) -> Dict[str, float]:
        head = str(getattr(self.cfg, "eval_head", "linear")).lower()
        ncm = None
        if head == "ncm":
            ncm = self._fit_ncm()

        cil_row, til_row = [], []
        for i in range(k + 1):
            task = self.tasks[i]
            loader = task_loader(task, self.cfg, train=False)
            res = evaluate_task(self.model, loader, self.seen_classes, task.classes,
                                self.device, self.cfg.time_steps,
                                encoding=getattr(self.cfg, "eval_encoding", "direct"),
                                passes=getattr(self.cfg, "eval_passes", 1),
                                calibrator=getattr(self.model, "calibrator", None),
                                ncm=ncm)
            cil_row.append(res["cil"])
            til_row.append(res["til"])
        self.cil_matrix.append(cil_row)
        self.til_matrix.append(til_row)
        return summarise([r[: i + 1] for i, r in enumerate(self.cil_matrix)],
                         [r[: i + 1] for i, r in enumerate(self.til_matrix)])

    def _fit_ncm(self):
        """Recompute class prototypes with the *current* feature extractor."""
        from .model import encode_direct
        from .ncm import NCMClassifier

        cfg = self.cfg
        examples, labels = self.model.buffer.get_all_data()
        if examples.numel() == 0:
            return None
        ncm = NCMClassifier.fit(
            self.model, examples, labels,
            n_classes=int(self.model.net.fc.out_features),
            encode_fn=lambda x: encode_direct(x, cfg.time_steps),
            temperature=float(getattr(cfg, "ncm_temperature", 10.0)),
            metric=str(getattr(cfg, "ncm_metric", "cosine")))
        self.last_ncm = ncm
        print(f"[ncm] {ncm.summary()}", flush=True)
        return ncm

    # ------------------------------------------------------------------ run
    def fit(self) -> Dict[str, object]:
        cfg = self.cfg
        print(f"[CLS-ER + SD^2-SNN] dataset={cfg.dataset} tasks={len(self.tasks)} "
              f"classes/task={cfg.classes_per_task} T={cfg.time_steps} width={cfg.width}")
        print(f"[model] params={self.model.num_params / 1e6:.2f}M "
              f"(+ 2 memories, {self.model.total_params / 1e6:.2f}M total)  "
              f"buffer={cfg.buffer_size}  reg_weight={cfg.reg_weight}  "
              f"eval_mode={cfg.eval_mode}")
        sd2 = {}
        if cfg.decompose:
            sd2["decompose"] = f"rank {cfg.rank_ratio}"
        for key, value in (("sparsity", cfg.sparse_weight), ("subspace", cfg.subspace_weight),
                           ("stability", cfg.stability_weight)):
            if value:
                sd2[key] = value
        print(f"[SD^2 regularisers] {sd2 if sd2 else 'none (CLS-ER + SNN adaptations only)'}")
        print(f"[SNN adaptations] selector={cfg.selector_mode}(β={cfg.selector_beta}), "
              f"consistency={cfg.consistency_loss}(τ={cfg.consistency_tau}), "
              f"ema_buffers={cfg.ema_buffers}, temporal={cfg.temporal_weight}, "
              f"proximal={cfg.proximal_weight}, calibrate={cfg.calibrate}, "
              f"replay_encoding={cfg.replay_encoding}")

        t0 = self._t0 = time.time()
        for k, task in enumerate(self.tasks):
            if k < self.start_task:
                continue
            print(f"\n===== task {k}: classes {task.classes} "
                  f"(train={len(task.train)}, test={len(task.test)}) =====", flush=True)
            if getattr(cfg, "calibrate", False):
                cal = self.model.calibrate_thresholds(task_loader(task, cfg, train=True))
                print(f"[calibrate] {cal['layers']} layers, mean firing rate "
                      f"{cal['rate_before']:.3f} -> {cal['rate_after']:.3f} "
                      f"(target {cfg.rate_target})", flush=True)
            stats = self.train_one_task(k)
            self.model.current_task = k
            self.seen_classes.extend(task.classes)
            info = self.model.end_of_task(k, task_loader(task, cfg, train=True),
                                          self.seen_classes, task.classes)

            summary = self.evaluate_all(k)
            record = {"task": k, "classes": task.classes, "train": stats, "eval": summary,
                      "post": info,
                      "elapsed_s": round(time.time() - t0 + self.elapsed_offset, 1)}
            self.history.append(record)
            print(f"[eval after task {k}] class-IL={summary['cil_acc']:.2f} "
                  f"task-IL={summary['til_acc']:.2f} F_cil={summary['forgetting_cil']:.2f} "
                  f"F_til={summary['forgetting_til']:.2f}", flush=True)
            self._save(record)
            if cfg.stop_after_task and (k + 1) >= cfg.stop_after_task:
                print(f"[early stop] stop_after_task={cfg.stop_after_task}", flush=True)
                break

        final = summarise([r[: i + 1] for i, r in enumerate(self.cil_matrix)],
                          [r[: i + 1] for i, r in enumerate(self.til_matrix)])
        final.update({"history": self.history, "config": cfg.to_dict(),
                      "buffer": self.model.buffer.state_dict()["num_seen_examples"],
                      "elapsed_s": round(time.time() - t0, 1)})
        if self.model.subspace is not None:
            final["subspace"] = self.model.subspace.summary()
        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(final, f, indent=2, default=float)
        print(f"\n[FINAL] class-IL = {final['cil_acc']:.2f}   task-IL = {final['til_acc']:.2f}   "
              f"forgetting(class-IL) = {final['forgetting_cil']:.2f}   "
              f"buffer seen = {final['buffer']}")
        print(f"[FINAL] saved {os.path.join(self.run_dir, 'results.json')}")
        return final

    # ------------------------------------------------------------- plumbing
    def _save(self, record: dict) -> None:
        if not self.cfg.save_ckpt:
            return
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.to_dict(),
                    "cil": self.cil_matrix, "til": self.til_matrix,
                    "task": record["task"]},
                   os.path.join(self.run_dir, "last.pt"))
        elapsed = (time.time() - self._t0 + self.elapsed_offset
                   if self._t0 is not None else self.elapsed_offset)
        with open(os.path.join(self.run_dir, "progress.json"), "w") as f:
            json.dump({"tasks_done": len(self.history), "n_tasks": len(self.tasks),
                       "cil_matrix": self.cil_matrix, "til_matrix": self.til_matrix,
                       "history": self.history,
                       "elapsed_s": round(elapsed, 1)},
                      f, indent=2, default=float)
