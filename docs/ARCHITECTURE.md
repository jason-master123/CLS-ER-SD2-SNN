# 架构说明：CLS-ER + SD²-SNN（SNN 原生版）

两份论文被拆开重组：

* **CLS-ER** 提供 *学习规则*（谁监督谁、何时更新记忆、怎么回放）；
* **SD²-SNN** 提供 *计算基底*（脉冲神经元、时序前向、放电率读出），外加三个可选正则；
* **本项目新增** 一层 *SNN 适配层*：把 CLS-ER 里隐含 ANN 假设的机制换成脉冲网成立的形式。

---

## 1. 总体数据流

```
                    ┌─────────────────────────────────────────┐
   当前任务批 ──────▶│ 直接编码 (T,B,3,32,32)                    │
                    └───────────────┬─────────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │ Spiking-ResNet-18（LIF × 17 层）          │
                    │  forward_temporal →                       │
                    │     logits   (B,K)   时间平均放电率读出    │
                    │     logits_t (T,B,K) 逐时刻证据（SNN 专属）│
                    │     feats    (T,B,C) 脉冲数 → 可靠性权重   │
                    └───────┬──────────────────────┬───────────┘
                            │                      │
        ┌───────────────────▼──────┐     ┌─────────▼─────────────┐
        │ L_CE（当前批 + 回放批）     │     │ 语义记忆（stable/plastic）│
        └───────────────────────────┘     │ 软门控 → z_sel          │
                                          │ 率域 KL → L_cons        │
                                          └─────────┬─────────────┘
                                                    │
        ┌───────────────────────────────────────────▼─────────────┐
        │ L = L_CE + λ_reg·L_cons + λ_t·L_temp + λ_p·L_prox        │
        │     (+ SD²-SNN 的 L_sparse / L_subspace / L_stab)        │
        └─────────────────────────────────────────────────────────┘
```

---

## 2. 为什么需要适配层（机制对照）

| # | CLS-ER 原机制 | ANN 隐含假设 | SNN 现实 | 本项目的替代 |
|---|---|---|---|---|
| ① | 只 EMA 参数 `W` | 函数仅由 W 决定 | 函数由 `(W, BN stats, v_th, τ)` 决定；BN 统计决定放电率分布 | **整状态 EMA**：参数 + buffers + 阈值 |
| ② | 逐样本硬选择记忆 | softmax 置信度可靠 | T 步平均放电率的方差 ~ r(1−r)/T，硬 argmax 不稳定 | **软门控** `w=σ(βΔc)`，β→∞ 退化回原版 |
| ③ | logits MSE | logits 连续、量级稳定 | 脉冲 logits 量纲小、随 T 变化，MSE 与尺度强耦合 | **率域温度 KL** + 脉冲数可靠性加权 |
| ④ | 单一输出监督 | 只有一个时刻的输出 | T 个时刻各自产生证据，"何时发放"是信息 | **时序蒸馏** `mean_t KL(p_t^T‖p_t^S)` |
| ⑤ | 无活动调节 | 激活不会死/饱和 | 放电率会崩塌或爆炸（SNN 特有失败模式） | **稳态阈值校准** |
| ⑥ | 仅 logits 蒸馏 | 梯度无偏 | 代理梯度有偏，参数在噪声方向漂移 | **近端约束** `λ‖θ_W−θ_stable‖²` |

实测证据（本仓库短跑）：

| 现象 | 原版 | 改造后 |
|---|---|---|
| `frac_stable`（stable 记忆被选中的比例） | **0.000**（恒定，stable 完全失效） | **0.629** |
| 任务开始时全网放电率 | 0.001（几乎静默） | 0.052（目标 0.05） |
| 学完一个任务后的放电率 | 0.385（过活跃，无处理） | 任务切换时校准回 0.089 |

---

## 3. 张量形状

| 阶段 | 形状 | 说明 |
|---|---|---|
| 数据批 | `(B, 3, 32, 32)` | 标准化 + 增广后 |
| 直接编码 | `(T, B, 3, 32, 32)` | 重复 T 次（确定性） |
| Poisson 编码 | `(T, B, 3, 32, 32)` | `Bernoulli(|x|)`，用于回放/随机评测 |
| conv/BN | `(T*B, C, H, W)` | 时间维摊入批维，BN 统计跨时间共享 |
| LIF | `(T, B, C, H, W)` | 逐步积分，模块内循环，无跨批状态 |
| 池化特征 `feats` | `(T, B, 512)` | 空间全局平均 |
| 率读出 `rate` | `(B, 512)` | `feats.mean(0)` = 平均放电率（Eq. 8） |
| `logits` | `(B, K)` | `fc(rate)` |
| `logits_t` | `(T, B, K)` | `fc(feats[t])`，时序蒸馏用 |
| 可靠性权重 | `(B,)` | 由 `feats.sum((0,2))` 得到 |

---

## 4. 三种记忆的更新语义（SNN 版）

```
global_step += 1
α_t = min(1 − 1/(global_step+1), α_max)      # 前期快、后期趋于 α_max
plastic ← α_t·plastic + (1−α_t)·working      # 概率 plastic_freq 触发
stable  ← α_t·stable  + (1−α_t)·working      # 概率 stable_freq 触发
```

**与原版的唯一区别**：EMA 的对象是**整状态**

```
参数      θ        ← α·θ + (1−α)·θ_work          （同原版）
BN 统计   μ,σ²     ← α·μ + (1−α)·μ_work          （新增）
阈值      v_th     ← α·v_th + (1−α)·v_th_work    （新增，标量）
```

这保证记忆是一个**行为上自洽的慢速模型**，而不是"旧权重 + 新统计"的错位组合。

---

## 5. 一致性损失的构造

```python
# 1) 记忆给出回放批的 logits
z_s = stable(spk_buf);  z_p = plastic(spk_buf)
c_s = softmax(z_s)[y];  c_p = softmax(z_p)[y]        # 真值类置信度

# 2) 软门控融合（β→∞ 即原版硬选择）
w_gate = sigmoid(β · (c_s − c_p))
z_sel  = w_gate · z_s + (1 − w_gate) · z_p

# 3) 逐样本权重 = 记忆分歧度 × 脉冲可靠性
w = |c_s − c_p| / max|·|  ·  clamp(n_spikes / (T·C·ρ), 0, 1)

# 4) 率域温度 KL（对尺度不敏感）
d = Σ_k p_sel,k · log(p_sel,k / p_student,k) · τ²
L_cons = Σ_i w_i·d_i / Σ_i w_i
```

对比原版：`L_cons = mean((net(buf) − z_sel.detach())²)`，`z_sel` 由硬 argmax 选出、等权平均。

---

## 6. 稳态阈值校准

```
for round in 1..R:
    for batch in first B batches of the task:
        前向，记录每层平均放电率 r_l
    for each LIF layer l:
        scale = clip( sqrt(r_l / ρ), 0.15, 8.0 )
        v_th_l ← v_th_l · scale
```

* 依据：放电率对阈值近似单调，`r ∝ 1/v_th` 的一阶近似给出平方根调节；
* 有界 + 多轮，避免单次调过头；
* 只作用于 working 模型，记忆通过整状态 EMA 跟随。

---

## 7. 模块地图

| 文件 | 职责 |
|---|---|
| `neurons.py` | LIF 动力学、ATan 替代梯度、放电率 hook |
| `backbone.py` | Spiking-ResNet-18；`forward` / `forward_temporal` |
| `buffer.py` | reservoir 采样（严格对齐官方 `utils/buffer.py`） |
| `memories.py` | 双记忆、软/硬选择器、整状态 EMA、近端项、评测融合 |
| `model.py` | `observe()` 总装；`calibrate_thresholds()`；`end_of_task()` |
| `sd2snn_extras.py` | Eq.(3)(13)(14)(16)(17)(18) 的脉冲正则实现 |
| `datasets.py` | 任务流；返回 `(x_aug, y, x_raw)` 三元组 |
| `metrics.py` | class-IL / task-IL / 遗忘 / BWT；多趟随机编码 |
| `trainer.py` | 逐任务训练、校准、评测矩阵、检查点 |

---

## 8. 与官方实现的逐行对应

| 官方 `models/clser.py` | 本项目 | 改动 |
|---|---|---|
| `deepcopy(self.net)` ×2 | `SemanticMemories.__init__` | 同 |
| `stable_model_prob[mask] > plastic_model_prob[mask]` | `select_logits` 的 `c_s > c_p` | 改为软门控，硬选择保留为选项 |
| `torch.where(sel_idx, z_s, z_p)` | `z_sel = w·z_s + (1−w)·z_p` | 改造 ② |
| `MSE(net(buf), z_sel.detach())` | `per_sample_distance(..., "kl_rate")` | 改造 ③ |
| `para.mul_(α).add_(1−α, param)` | `_ema_into`（参数 + buffers + 阈值） | 改造 ① |
| `buffer.add_data(not_aug, labels)` | `add_data(x_raw, y)` | 同 |
| `torch.cat((inputs, buf_inputs))` | 同（一次前向） | 同 |
| — | `calibrate_thresholds` | 新增 ⑤ |
| — | `L_temp`（逐时刻 KL） | 新增 ④ |
| — | `L_prox` | 新增 ⑥ |

---

## 9. 计算开销

| 项 | 数值（width=32, T=4） |
|---|---|
| 骨干参数 | ≈ 2.4 M |
| 每步前向 | 2 次（当前批 + 回放批，合并成一次调用 → 实际 1 次批量前向 + 1 次记忆前向） |
| 记忆显存 | 2 × 骨干（约 20 MB）+ buffer（500×3×32×32 ≈ 6 MB） |
| 时序蒸馏额外开销 | 一次 `fc`（可忽略）；时序教师需一次记忆前向（已在选择器里算过，可复用） |
| 稳态校准 | 每任务约 8 个 batch 的前向（可忽略） |
| 推理（默认 `--eval-mode working`） | 与单模型相同 |
