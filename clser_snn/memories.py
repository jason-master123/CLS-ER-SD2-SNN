"""双语义记忆 —— 针对脉冲网络的改造版（brain: 海马体 / 新皮层）

原版 CLS-ER 的三条机制都隐含 ANN 假设，直接搬到 SNN 上会失效，这里逐条替换：

=====================================  ======================================
ANN 机制（失效原因）                     SNN 改造
=====================================  ======================================
只 EMA 参数 (W)                         **整状态 EMA**：参数 + BN running stats
                                        + 神经元阈值。脉冲网的"函数"由
                                        (W, BN 统计, v_th, τ) 共同决定，只复制 W
                                        会得到一个放电率分布错误的"记忆"。
按真值类置信度硬选 (per-exemplar argmax)  **软门控**：w=σ(β·(c_s−c_p)) 后加权融合。
                                        T 步的放电率估计方差 ~ r(1−r)/T，硬选择
                                        在少量脉冲下极不稳定（实测恒选 plastic）。
                                        β→∞ 时退化为原版硬选择。
logits 上的 MSE                          **率域 KL + 可靠性加权**：脉冲 logits 是
                                        放电率的线性映射，量纲小、方差大；
                                        改为温度 KL 并按样本脉冲数降权。
=====================================  ======================================

另外提供脉冲网特有的项：时序一致性（逐 timestep 的 logits 蒸馏）与近端约束
（把 working 拉向 stable 的参数空间），见 ``extra_losses``。
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMemories(nn.Module):
    def __init__(self, working: nn.Module, cfg) -> None:
        super().__init__()
        # LIF nodes cache their last membrane for the ``membrane`` readout; that
        # tensor may still hold an autograd graph, and non-leaf tensors cannot be
        # deep-copied. Detach the caches before duplicating the working model.
        for module in working.modules():
            if hasattr(module, "clear_cache"):
                module.clear_cache()
        self.plastic = copy.deepcopy(working)
        self.stable = copy.deepcopy(working)
        for m in (self.plastic, self.stable):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.plastic_freq = float(cfg.plastic_model_update_freq)
        self.plastic_alpha = float(cfg.plastic_model_alpha)
        self.stable_freq = float(cfg.stable_model_update_freq)
        self.stable_alpha = float(cfg.stable_model_alpha)

        # SNN 改造 1：整状态 EMA（BN 统计量 + 阈值都在 buffers/属性里）
        self.ema_buffers = bool(getattr(cfg, "ema_buffers", True))

        # SNN 改造 2：软门控
        self.selector_mode = getattr(cfg, "selector_mode", "soft")
        self.selector_beta = float(getattr(cfg, "selector_beta", 4.0))
        self.selector_reliability = bool(getattr(cfg, "selector_reliability", True))

        self.global_step = 0
        self.stats: Dict[str, float] = {}
        self.eval()

    # ------------------------------------------------------------- selection
    def _memory_logits(self, which: str, x: torch.Tensor) -> torch.Tensor:
        """Logits of one memory copy, honouring the training-time head mode.

        With ``--head-mode ncm`` the linear layers of the memories are unused (they
        never receive gradient), so the memories must be scored with **their own**
        class means — each memory lives in its own feature space.
        """
        net = self.stable if which == "stable" else self.plastic
        heads = getattr(self, "heads", None)
        if heads is None:
            return net(x)
        feats = net.features(x).mean(dim=0)
        return heads[which](feats)

    @torch.no_grad()
    def select_logits(self, x: torch.Tensor, labels: torch.Tensor,
                      reliability: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CLS-ER Eq.(1) 的选择步骤，改为软门控。

        Returns ``(z_selected, per_sample_weight)``; both detached. ``z_selected``
        在 hard 模式下就是原版的 ``torch.where``，在 soft 模式下是两记忆的凸组合。
        """
        z_s = self._memory_logits("stable", x)
        z_p = self._memory_logits("plastic", x)
        p_s = F.softmax(z_s, dim=1)
        p_p = F.softmax(z_p, dim=1)

        idx = labels.unsqueeze(1)
        c_s = p_s.gather(1, idx).squeeze(1)          # 真值类置信度
        c_p = p_p.gather(1, idx).squeeze(1)

        if self.selector_mode == "hard":
            use_stable = (c_s > c_p).float()
            z_sel = torch.where(use_stable.unsqueeze(1).bool(), z_s, z_p)
            w = torch.ones_like(c_s)
        else:
            # 软门控：置信度差越大越倾向某一侧，β 控制锐度
            gate_stable = torch.sigmoid(self.selector_beta * (c_s - c_p))
            z_sel = gate_stable.unsqueeze(1) * z_s + (1 - gate_stable.unsqueeze(1)) * z_p
            # 该样本的"可选择性"：两个记忆意见越一致，约束信息越少 → 降权
            w = (c_s - c_p).abs()
            w = w / (w.max() + 1e-6)

        if reliability is not None and self.selector_reliability:
            w = w * reliability.clamp(0.0, 1.0)

        self.stats["frac_stable"] = float((c_s > c_p).float().mean())
        self.stats["mean_margin"] = float((c_s - c_p).abs().mean())
        return z_sel.detach(), w.detach()

    # ------------------------------------------------------- inference-time
    @torch.no_grad()
    def stable_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """逐 timestep 的稳定记忆 logits（供时序蒸馏使用）。"""
        heads = getattr(self, "heads", None)
        if hasattr(self.stable, "forward_temporal"):
            _, logits_t, feats = self.stable.forward_temporal(x)
            if heads is not None:                       # ncm 头：逐时刻重新打分
                return heads["stable"](feats)
            return logits_t
        return self.stable(x).unsqueeze(0)

    @torch.no_grad()
    def fused_logits(self, x: torch.Tensor, working_logits: torch.Tensor,
                     mode: str = "working") -> torch.Tensor:
        if mode == "working":
            return working_logits
        if mode == "stable":
            return self._memory_logits("stable", x)
        if mode == "plastic":
            return self._memory_logits("plastic", x)
        if mode == "mean":
            probs = [F.softmax(working_logits, 1),
                     F.softmax(self._memory_logits("stable", x), 1),
                     F.softmax(self._memory_logits("plastic", x), 1)]
            return torch.stack(probs).mean(0)
        if mode == "maxconf":
            logits = torch.stack([working_logits, self._memory_logits("stable", x),
                                  self._memory_logits("plastic", x)])
            conf = F.softmax(logits, dim=-1).max(dim=-1).values
            pick = conf.argmax(dim=0)
            return logits.gather(0, pick.view(1, -1, 1).expand(1, -1, logits.shape[-1])).squeeze(0)
        raise ValueError(f"unknown eval_mode {mode}")

    # ------------------------------------------------------------- updating
    @torch.no_grad()
    def update(self, working: nn.Module) -> None:
        self.global_step += 1
        if torch.rand(1).item() < self.plastic_freq:
            self._ema_into(self.plastic, working, self.plastic_alpha)
        if torch.rand(1).item() < self.stable_freq:
            self._ema_into(self.stable, working, self.stable_alpha)

    def _ema_into(self, memory: nn.Module, working: nn.Module, alpha_max: float) -> None:
        alpha = min(1.0 - 1.0 / (self.global_step + 1), alpha_max)
        for ema_p, p in zip(memory.parameters(), working.parameters()):
            ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
        if not self.ema_buffers:
            return
        # --- SNN 改造 1（续）：脉冲网的 BN 统计量与阈值必须一起搬家，
        #     否则记忆看到的是"旧权重 + 新统计"，放电率分布会漂移。
        mem_buffers = dict(memory.named_buffers())
        for name, b in working.named_buffers():
            mb = mem_buffers.get(name)
            if mb is None or mb.shape != b.shape:
                continue
            if mb.dtype.is_floating_point:
                mb.data.mul_(alpha).add_(b.data, alpha=1.0 - alpha)
            else:
                mb.data.copy_(b.data)
        # 阈值是普通属性而非 buffer：按同一 alpha 做标量 EMA。
        # 可学习的阈值已经是一个 nn.Parameter，上面那段参数 EMA 会自动处理它——
        # 再走一次会让它被 EMA 两遍，所以这里跳过。
        for m_mem, m_work in zip(memory.modules(), working.modules()):
            if not (hasattr(m_work, "v_threshold") and hasattr(m_mem, "v_threshold")):
                continue
            if getattr(m_work, "learnable_vth", False):
                continue
            m_mem.v_threshold = (alpha * m_mem.v_threshold
                                 + (1 - alpha) * m_work.v_threshold)

    # -------------------------------------------------- 近端约束（新皮层固化）
    def proximal_term(self, working: nn.Module) -> torch.Tensor:
        """λ_p‖θ_W − θ_stable‖²：把 working 拉向长期记忆的参数流形。

        脉冲网的代理梯度有偏，纯 CE 更新时参数会在噪声方向上漂移；近端项相当于
        新皮层对海马体快速编码的"慢约束"，比只靠 logits 蒸馏更直接。
        """
        total = None
        for p, s in zip(working.parameters(), self.stable.parameters()):
            term = (p - s.detach()).pow(2).sum()
            total = term if total is None else total + term
        return total if total is not None else torch.zeros(())

    # ---------------------------------------------------------------- utils
    def train(self, mode: bool = True):        # type: ignore[override]
        super().train(False)                   # 记忆永远 eval
        self.plastic.eval()
        self.stable.eval()
        return self

    def synchronize(self, working: nn.Module) -> None:
        with torch.no_grad():
            for memory in (self.plastic, self.stable):
                for ema_p, p in zip(memory.parameters(), working.parameters()):
                    ema_p.data.copy_(p.data)
                mem_buffers = dict(memory.named_buffers())
                for name, b in working.named_buffers():
                    if name in mem_buffers and mem_buffers[name].shape == b.shape:
                        mem_buffers[name].data.copy_(b.data)
