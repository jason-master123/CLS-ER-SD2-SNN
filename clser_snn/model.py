"""混合模型：CLS-ER 的学习规则 × 脉冲网络的原生改造。

一次优化步（在原版 ``CLSER.observe`` 的基础上做了 SNN 适配，逐条标注）：

1. 取当前任务批 + 回放批；回放批可选用**随机编码**（``--replay-encoding poisson``），
   让回放覆盖更宽的输入分布 —— 脉冲网对精确脉冲时刻敏感，确定性重复回放的
   信息量比 ANN 更低。
2. 一次前向得到 ``logits``、**逐 timestep logits** 与特征（(T,B,C)）。
3. ``L_CE``：当前批与回放批的联合交叉熵（与 CLS-ER 相同）。
4. 双语义记忆在回放批上给出 logits，**软门控**融合得到伪标签 ``z_sel``（SNN 改造：
   硬选择在 T 步放电率估计上方差过大，实测会退化成恒选一个记忆）。
5. ``L_cons``：**率域 KL**（温度 τ，对量纲不敏感）而非 logits MSE，并按样本的
   脉冲数做**可靠性加权**（脉冲太少的样本不参与蒸馏）。
6. 可选：**时序蒸馏**（逐 timestep logits 的 KL，ANN 版没有这一项）、
   **近端约束**（‖θ_W−θ_stable‖²）、以及 SD²-SNN 的三个正则。
7. 反向 / 更新 / 写入 reservoir / 两次概率触发 EMA（整状态：参数 + BN 统计 + 阈值）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .buffer import ReservoirBuffer
from .calibration import LogitCalibrator, build_calibrator, install_cosine_head
from .memories import SemanticMemories
from .neurons import RateMonitor
from .sd2snn_extras import (
    SparseSubspaceMemory,
    basis_snapshot,
    layer_sparsity,
    parameter_groups,
    replace_with_factorized,
    stability_penalty,
)


def encode_direct(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Deterministic direct encoding: repeat the frame T times."""
    return x.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)


def encode_poisson(x: torch.Tensor, time_steps: int, generator: Optional[torch.Generator] = None
                   ) -> torch.Tensor:
    """Stochastic rate coding: Bernoulli spikes with rate |x| (used for replay)."""
    rate = x.clamp(0.0, 1.0).unsqueeze(0)
    probs = rate.expand(time_steps, *x.shape)
    return (torch.rand(probs.shape, device=x.device, generator=generator) < probs).to(x.dtype)


def encode_input(x: torch.Tensor, time_steps: int, mode: str = "direct",
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return encode_poisson(x, time_steps, generator) if mode == "poisson" \
        else encode_direct(x, time_steps)


def _amp_ctx(enabled: bool):
    """bf16 autocast when requested (fp16 was unsafe for this network)."""
    if not enabled:
        return torch.autocast(device_type="cuda", enabled=False)
    try:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def per_sample_distance(student: torch.Tensor, teacher: torch.Tensor,
                        mode: str = "kl_rate", tau: float = 2.0) -> torch.Tensor:
    """Per-example discrepancy between two logit vectors -> shape (B,).

    ``kl_rate`` / ``kl``  — 温度 KL（对 logit 尺度不敏感，脉冲网首选）
    ``mse``               — CLS-ER Eq.(1) 的原始形式（ANN 口径）
    """
    if mode == "mse":
        return (student - teacher).pow(2).mean(dim=1)
    t = max(1e-3, tau)
    p_t = F.softmax(teacher / t, dim=1).clamp_min(1e-8)
    log_p_s = F.log_softmax(student / t, dim=1)
    return (p_t * (p_t.log() - log_p_s)).sum(dim=1) * (t * t)


class SpikingCLSER(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.net = build_backbone(cfg).to(self.device)
        if getattr(cfg, "decompose", False):
            replace_with_factorized(self.net, cfg.rank_ratio)
        if getattr(cfg, "cosine_head", False):
            install_cosine_head(self, getattr(cfg, "cosine_scale", 10.0))

        self.monitor = RateMonitor().attach(self.net)
        self.memories = SemanticMemories(self.net, cfg).to(self.device)
        self.buffer = ReservoirBuffer(cfg.buffer_size, device=str(self.device))
        self.subspace = SparseSubspaceMemory(cfg.subspace_eps) \
            if getattr(cfg, "subspace_weight", 0.0) > 0 else None
        # post-hoc logit calibration, refitted at every task boundary
        self.calibrator = LogitCalibrator(n_classes=int(self.net.fc.out_features))
        # feature-space class prototypes: a *training* objective when either the
        # attraction (proto_weight) or the inter-class hinge (proto_margin_weight)
        # is enabled — both need the prototypes, so create them for either.
        self.prototypes = None
        if (getattr(cfg, "proto_weight", 0.0) > 0
                or getattr(cfg, "proto_margin_weight", 0.0) > 0):
            from .prototypes import PrototypeMemory
            dim = int(getattr(self.net.fc, "in_features", 0)) or None
            if dim is None:                     # CosineHead exposes in_features too
                raise ValueError("cannot infer feature dimension for prototypes")
            self.prototypes = PrototypeMemory(
                n_classes=int(self.net.fc.out_features), dim=dim,
                temperature=float(getattr(cfg, "proto_temp", 0.1)),
                device=str(self.device))

        # ---- classifier used *during training* -------------------------------
        # ``linear``: the usual learnable head (net.fc).
        # ``ncm``   : a parameter-free class-mean head rebuilt from the buffer; the
        #             linear layer is then unused, which removes the new-class logit
        #             bias at its source instead of correcting it after the fact.
        self.head_mode = str(getattr(cfg, "head_mode", "linear")).lower()
        self.heads = None
        if self.head_mode == "ncm":
            from .meanhead import HeadBank
            dim = int(getattr(self.net.fc, "in_features", 0))
            self.heads = HeadBank(
                n_classes=int(self.net.fc.out_features), dim=dim,
                metric=str(getattr(cfg, "head_metric", "cosine")),
                temperature=float(getattr(cfg, "head_temperature", 10.0))).to(self.device)
        elif self.head_mode != "linear":
            raise ValueError(f"unknown head_mode {self.head_mode!r}")

        self.current_task = 0
        # the memories must score themselves with their own class means when the
        # ncm head is active (their linear layers never receive gradient)
        if self.heads is not None:
            self.memories.heads = self.heads
        self.prev_basis: Optional[dict] = None
        self.last_stats: Dict[str, float] = {}
        self._replay_gen = torch.Generator(device=str(self.device)).manual_seed(cfg.seed + 7)
        self.opt = self.build_optimizer(cfg)

    # ---------------------------------------------------------------- utils
    @property
    def num_params(self) -> int:
        """Trainable backbone size (the two semantic memories are frozen copies)."""
        return sum(p.numel() for p in self.net.parameters())

    @property
    def total_params(self) -> int:
        """Everything kept in memory: working model + stable + plastic."""
        return sum(p.numel() for p in self.parameters())

    def build_optimizer(self, cfg) -> torch.optim.Optimizer:
        groups = parameter_groups(self.net, cfg.lr, cfg.weight_decay,
                                  getattr(cfg, "u_lr_ratio", 0.1))
        if cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(groups)
        if cfg.optimizer.lower() == "sgd":
            return torch.optim.SGD(groups,
                                   momentum=float(getattr(cfg, "sgd_momentum", 0.9)))
        raise ValueError(cfg.optimizer)

    def set_optimizer(self, opt) -> None:
        self.opt = opt

    def train_step_mode(self, train: bool = True) -> None:
        self.net.train(train)
        # 官方 CLS-ER 从不对两个记忆模型调用 .eval()：它们的 deepcopy 发生在
        # net 还是 train 模式时，所以记忆的前向用 **batch 统计量**做 BN，只需要
        # EMA 参数就够了。我们的记忆默认在 eval 模式（用 running stats），
        # 因此必须整状态 EMA。``--memory-train-mode`` 还原官方行为，用于复现。
        if getattr(self.cfg, "memory_train_mode", False):
            self.memories.train(train)
        else:
            self.memories.eval()

    def _reliability(self, feats: torch.Tensor) -> torch.Tensor:
        """每样本的脉冲可信度：脉冲总数相对期望值的比例（0..1）。

        脉冲网在 T 很小时，放电率估计的方差是 O(r(1-r)/T)；脉冲过少的样本
        其 memory logits 基本是噪声，蒸馏时应降权。

        只在 ``readout="rate"`` 时才有意义：膜电位（有正有负、经 LayerNorm）与
        稠密特征（无脉冲概念）都算不出"脉冲数"，此时一律返回 1，让一致性项保持
        原始强度——否则加权会把 CLS-ER 的约束意外关掉，混淆对照。
        """
        if (str(getattr(self.cfg, "readout", "rate")).lower() != "rate"
                or str(getattr(self.cfg, "backbone", "spiking")).lower() != "spiking"):
            return torch.ones(feats.shape[1], device=feats.device, dtype=feats.dtype)
        n_spikes = feats.sum(dim=(0, 2))                      # (B,)
        expected = max(1e-6, feats.shape[0] * feats.shape[2] * float(self.cfg.rate_target))
        return (n_spikes / expected).clamp(0.0, 1.0)

    # -------------------------------------------------------------- forward
    def forward(self, x_raw: torch.Tensor, augment_fn=None) -> torch.Tensor:
        x = augment_fn(x_raw) if augment_fn is not None else x_raw
        spk = encode_direct(x, self.cfg.time_steps)
        logits = self.net(spk)
        if self.cfg.eval_mode == "working":
            return logits
        return self.memories.fused_logits(spk, logits, self.cfg.eval_mode)

    # -------------------------------------------------------------- observe
    def observe(self, x_aug: torch.Tensor, y: torch.Tensor,
                x_raw: torch.Tensor, augment_fn=None) -> Dict[str, float]:
        cfg = self.cfg
        self.train_step_mode(True)
        T = cfg.time_steps

        parts_x, parts_y = [x_aug], [y]
        n_stream = x_aug.shape[0]

        buf_x = buf_y = None
        if not self.buffer.is_empty():
            buf_raw, buf_y = self.buffer.get_data(cfg.minibatch_size)
            buf_x = augment_fn(buf_raw) if augment_fn is not None else buf_raw
            parts_x.append(buf_x)
            parts_y.append(buf_y)

        x_all = torch.cat(parts_x, dim=0)
        y_all = torch.cat(parts_y, dim=0)

        with _amp_ctx(bool(getattr(cfg, "amp", False)) and self.device.type == "cuda"):
            # 流式数据用确定性直接编码；回放可用随机编码（SNN 改造 6）
            spk_stream = encode_direct(x_all[:n_stream], T)
            if buf_x is not None:
                spk_buf = encode_input(buf_x, T, getattr(cfg, "replay_encoding", "direct"),
                                       self._replay_gen)
                spk_all = torch.cat([spk_stream, spk_buf], dim=1)
            else:
                spk_all = spk_stream

            logits_all, logits_t_all, feats = self.net.forward_temporal(spk_all)
            rates = dict(self.monitor.rates)

            if self.heads is not None:
                # score with the class-mean head instead of the (unused) linear layer
                logits_all = self.heads["working"](feats.mean(dim=0))
                logits_t_all = self.heads["working"](feats)

            l_ce = F.cross_entropy(logits_all, y_all)

            # ---- CLS-ER 一致性项（SNN 改造 2/3：软门控 + 率域 KL + 可靠性加权）
            l_cons = torch.zeros((), device=self.device)
            l_temp = torch.zeros((), device=self.device)
            if buf_x is not None:
                spk_buf_only = spk_all[:, n_stream:]
                reliability = self._reliability(feats[:, n_stream:])
                z_sel, w_sel = self.memories.select_logits(spk_buf_only, buf_y, reliability)

                z_student = logits_all[n_stream:]
                dist = per_sample_distance(z_student, z_sel,
                                           cfg.consistency_loss, cfg.consistency_tau)
                weights = w_sel.clamp_min(1e-3)
                l_cons = (dist * weights).sum() / weights.sum()

                # SNN 改造 4：时序蒸馏（逐时刻 KL + soft-DTW 轨迹对齐）
                if getattr(cfg, "temporal_weight", 0.0) > 0:
                    mode = str(getattr(cfg, "temporal_mode", "step")).lower()
                    s_logits_t = logits_t_all[:, n_stream:]          # (T, B, K)
                    tau = max(1e-3, cfg.temporal_tau)
                    terms = []
                    if mode in ("step", "both"):
                        t_logits_t = self.memories.stable_temporal(encode_direct(buf_x, T))
                        p_t = F.softmax(t_logits_t.detach() / tau, dim=-1).clamp_min(1e-8)
                        log_p_s = F.log_softmax(s_logits_t / tau, dim=-1)
                        kl_t = (p_t * (p_t.log() - log_p_s)).sum(-1) * (tau * tau)   # (T, B)
                        terms.append(kl_t.mean())
                    if mode in ("dtw", "both"):
                        # 同一批回放图像在**另一个时间分辨率**下的记忆轨迹；soft-DTW
                        # 允许学生的证据整体早/晚出现，只惩罚形状不符（时间平移不算遗忘）
                        from .temporal import temporal_alignment_loss
                        gamma = float(getattr(cfg, "temporal_gamma", 1.0))
                        scales = [float(s) for s in
                                  str(getattr(cfg, "temporal_scales", "0.5")).split(",") if s.strip()]
                        student_traj = s_logits_t.permute(1, 0, 2)   # (B, T, K)
                        for scale in scales:
                            T2 = max(2, int(round(T * scale)))
                            teacher2 = self.memories.stable_temporal(encode_direct(buf_x, T2))
                            teacher_traj = teacher2.detach().permute(1, 0, 2)   # (B, T2, K)
                            terms.append(temporal_alignment_loss(student_traj, teacher_traj, gamma))
                    if terms:
                        l_temp = torch.stack(terms).mean()

            loss = l_ce + cfg.reg_weight * l_cons
            if getattr(cfg, "temporal_weight", 0.0) > 0:
                loss = loss + cfg.temporal_weight * l_temp

            # ---- 特征侧约束：把样本（当前 + 回放）拉向自己的类原型
            #      NCM 实验表明瓶颈在"特征没有跨任务结构"，这条损失直接针对它。
            l_proto = torch.zeros((), device=self.device)
            l_margin = torch.zeros((), device=self.device)
            if self.prototypes is not None:
                # the classifier consumes ``rate`` = time-averaged features
                rate_all = feats.mean(dim=0)
                l_proto = self.prototypes.loss(rate_all, y_all)
                loss = loss + getattr(cfg, "proto_weight", 0.0) * l_proto
                if getattr(cfg, "proto_margin_weight", 0.0) > 0:
                    # 只有类内吸引、没有类间排斥时，簇只是各自收紧，
                    # 相对布局照样乱 —— 这一项补上"negative"。
                    l_margin = self.prototypes.margin_loss(
                        rate_all, y_all, margin=float(getattr(cfg, "proto_margin", 0.1)))
                    loss = loss + cfg.proto_margin_weight * l_margin

            # SNN 改造 5：近端约束（把 working 拉向 stable 的参数流形）
            l_prox = torch.zeros((), device=self.device)
            if getattr(cfg, "proximal_weight", 0.0) > 0:
                l_prox = self.memories.proximal_term(self.net)
                loss = loss + cfg.proximal_weight * l_prox

            # SD^2-SNN 可选正则
            l_sparse = torch.zeros((), device=self.device)
            l_subspace = torch.zeros((), device=self.device)
            l_stab = torch.zeros((), device=self.device)
            if getattr(cfg, "sparse_weight", 0.0) > 0 and rates:
                l_sparse = layer_sparsity(rates, cfg.sparse_target)
                loss = loss + cfg.sparse_weight * l_sparse
            if self.subspace is not None and self.subspace.entries and rates:
                l_subspace = self.subspace.loss(rates)
                loss = loss + cfg.subspace_weight * l_subspace
            if getattr(cfg, "stability_weight", 0.0) > 0 and self.prev_basis is not None:
                l_stab = stability_penalty(self.net, self.prev_basis)
                loss = loss + cfg.stability_weight * l_stab

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=x_raw.detach(), labels=y)
        self.memories.update(self.net)
        self.monitor.clear()

        self.last_stats = {
            "loss": float(loss.detach()),
            "ce": float(l_ce.detach()),
            "cons": float(l_cons.detach()),
            "cons_w": float(cfg.reg_weight * l_cons.detach()),
            "temp": float(l_temp.detach()),
            "prox": float(l_prox.detach()),
            "proto": float(l_proto.detach()),
            "margin": float(l_margin.detach()),
            "sparse": float(l_sparse.detach()),
            "subspace": float(l_subspace.detach()),
            "stability": float(l_stab.detach()),
            "frac_stable": self.memories.stats.get("frac_stable", float("nan")),
            "sel_margin": self.memories.stats.get("mean_margin", float("nan")),
        }
        return self.last_stats

    # ------------------------------------------------- 原型刷新（特征侧约束）
    @torch.no_grad()
    def refresh_prototypes(self) -> Optional[str]:
        """Rebuild the class prototypes from the buffer with the current model.

        Called at every epoch boundary and at the start of every task. Exact
        re-computation (instead of an EMA) is essential: normalised feature means
        taken from different, fast-rotating feature spaces cancel out.
        """
        if self.prototypes is None or self.buffer.is_empty():
            return None
        examples, labels = self.buffer.get_all_data()
        n = self.prototypes.refresh(
            self, examples, labels,
            encode_fn=lambda x: encode_direct(x, self.cfg.time_steps),
            min_samples=int(getattr(self.cfg, "proto_min_samples", 4)))
        return self.prototypes.summary() if n else None

    # --------------------------------------------- 类均值头刷新（ncm 训练模式）
    @torch.no_grad()
    def refresh_heads(self, names=None) -> Optional[str]:
        """Rebuild the class means used by ``--head-mode ncm`` from the buffer.

        Called at the same boundaries as ``refresh_prototypes``. Each model copy
        (working / stable / plastic) gets **its own** prototypes, because each has
        its own feature space; sharing them would make the consistency term
        compare scores from two different geometries.
        """
        if self.heads is None or self.buffer.is_empty():
            return None
        from .meanhead import fit_heads
        info = fit_heads(self.heads, self, self.buffer,
                         lambda x: encode_direct(x, self.cfg.time_steps),
                         names=names)
        if not info:
            return None
        return ", ".join(f"{k}:{v}" for k, v in info.items())

    # --------------------------------------------------- 稳态阈值校准（SNN 特有）
    @torch.no_grad()
    def calibrate_thresholds(self, loader) -> Dict[str, float]:
        """按层测量平均放电率，用平方根比例调整阈值，把活动拉回目标区间。

        脉冲持续学习里常见"放电率崩塌/爆炸"：新任务把膜电位推向饱和或静默。
        生物上用稳态可塑性（homeostatic plasticity）调节兴奋性，这里用同样的
        思路改 v_threshold，而不是让稀疏正则去硬压。默认做 ``calibrate_rounds``
        轮（每轮的缩放幅度有界），避免一次调过头。
        """
        cfg = self.cfg
        rounds = max(1, int(getattr(cfg, "calibrate_rounds", 2)))
        first_before = None
        before: list = []
        after: list = []
        for _ in range(rounds):
            self.train_step_mode(False)
            acc: Dict[str, torch.Tensor] = {}
            seen = 0
            for x_aug, _y, x_raw in loader:
                spk = encode_direct(x_raw.to(self.device), cfg.time_steps)
                self.net(spk)
                for name, r in self.monitor.rates.items():
                    acc[name] = acc.get(name, 0.0) + r.detach().mean(dim=0)
                seen += 1
                if seen >= cfg.calibrate_batches:
                    break
            self.monitor.clear()
            if not acc:
                break
            before, after = [], []
            lifs = dict(self.net.named_modules())
            for name, rate in acc.items():
                mean_rate = float((rate / max(1, seen)).mean())
                before.append(mean_rate)
                target = float(cfg.rate_target)
                scale = max(0.15, min(8.0, (mean_rate / max(target, 1e-6)) ** 0.5))
                module = lifs.get(name)
                if module is not None and hasattr(module, "v_threshold"):
                    module.v_threshold = float(module.v_threshold * scale)
                after.append(mean_rate / scale)
            if first_before is None:
                first_before = sum(before) / len(before)
        if not before:
            return {"layers": 0, "rate_before": float("nan"), "rate_after": float("nan")}
        return {"layers": len(before), "rounds": rounds,
                "rate_before": first_before,
                "rate_mid": sum(before) / len(before),
                "rate_after": sum(after) / len(after)}

    # --------------------------------------------------------- 任务边界
    @torch.no_grad()
    def end_of_task(self, task: int, train_loader,
                    seen_classes: Optional[list] = None,
                    task_classes: Optional[list] = None) -> Dict[str, object]:
        info: Dict[str, object] = {}
        if self.subspace is not None:
            self.train_step_mode(False)
            acc: Dict[str, torch.Tensor] = {}
            n = 0
            for _x_aug, _y, x_raw in train_loader:
                spk = encode_direct(x_raw.to(self.device), self.cfg.time_steps)
                self.net(spk)
                for name, r in self.monitor.rates.items():
                    acc[name] = acc.get(name, 0.0) + r.detach().sum(dim=0)
                n += x_raw.shape[0]
                if n >= 4096:
                    break
            self.subspace.update(task, {k: v / max(1, n) for k, v in acc.items()})
            info["subspace"] = self.subspace.summary()
        if getattr(self.cfg, "decompose", False):
            self.prev_basis = basis_snapshot(self.net)

        # ---- post-hoc logit calibration, refitted for the classes seen so far
        if str(getattr(self.cfg, "calib_mode", "none")).strip().lower() not in ("", "none"):
            from .datasets import aug_transform
            self.train_step_mode(False)
            aug = aug_transform() if getattr(self.cfg, "calib_augment", True) else None
            encode_fn = lambda x: encode_direct(  # noqa: E731
                aug(x) if aug is not None else x, self.cfg.time_steps)
            self.calibrator = build_calibrator(
                self.cfg, self, self.buffer,
                seen_classes or [], task_classes or [],
                encode_fn=encode_fn, augment_fn=aug)
            info["calibration"] = self.calibrator.summary()
            if self.calibrator.info:
                info["calibration_info"] = {k: round(float(v), 4)
                                            for k, v in self.calibrator.info.items()}
            print(f"[calib] {self.calibrator.summary()}", flush=True)

        self.monitor.clear()
        return info

    # ------------------------------------------------------------- ckpt i/o
    def state_dict(self, *args, **kwargs):        # noqa: D401
        sd = {"net": self.net.state_dict(),
              "stable": self.memories.stable.state_dict(),
              "plastic": self.memories.plastic.state_dict(),
              "buffer": self.buffer.state_dict(),
              "calibrator": self.calibrator.state_dict(),
              "global_step": self.memories.global_step,
              "task": self.current_task}
        if self.prototypes is not None:
            sd["prototypes"] = self.prototypes.state_dict()
        return sd

    def load_state_dict(self, sd, strict: bool = True):   # noqa: D401
        self.net.load_state_dict(sd["net"])
        self.memories.stable.load_state_dict(sd["stable"])
        self.memories.plastic.load_state_dict(sd["plastic"])
        self.buffer.load_state_dict(sd["buffer"])
        if "calibrator" in sd:
            self.calibrator = LogitCalibrator.from_state(sd["calibrator"])
        if self.prototypes is not None and "prototypes" in sd:
            from .prototypes import PrototypeMemory
            self.prototypes = PrototypeMemory.from_state(sd["prototypes"])
            self.prototypes.to(self.device)
        self.memories.global_step = sd.get("global_step", 0)
        self.current_task = sd.get("task", 0)
