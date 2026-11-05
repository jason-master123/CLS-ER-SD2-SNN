# 复现指南

所有数字都来自单张消费级 GPU（8 GB）上的单 seed 训练。每组运行的**完整配置**在
`runs/<run_name>/config.json`，逐任务评测矩阵在 `progress.json`，
最终指标在 `results.json`。

## 环境

```
Python 3.13 / PyTorch 2.x（CUDA 12.x）
GPU：8 GB（实测峰值显存 3.0–5.0 GB）
```

```bash
python -m pip install -r requirements.txt
python tests/smoke_test.py          # 35 项自检，CPU，< 3 分钟
```

数据集：优先 `./data`；否则复用 `../CLS-ER/datasets/CIFAR10`、
`../SD2-SNN-repro/data/c100root`；再否则由 torchvision 下载。

## 主结果（5 任务 × 25 epochs，batch 32，buffer 500，width 32，seed 0）

下表每一行都可用对应命令重新得到。**注意 `--tag` 会决定输出目录名，若要和报告里的
数字逐位对齐，请保留原 tag。**

| 结果 | class-IL | task-IL | 耗时 | 关键开关 |
|---|---|---|---|---|
| `snn_e25`（基线） | 57.51 | 88.98 | 108 min | 无 |
| `ann_e25`（ANN 口径） | 39.50 | 80.36 | 126 min | `--selector-mode hard --consistency-loss mse --no-ema-buffers` |
| `ann_ctrl`（稠密对照） | 65.56 | 92.99 | 71 min | `--backbone ann` |
| `snn_plif` | 59.66 | 91.55 | 163 min | `--lif-learnable` |
| `snn_t8` | 58.80 | 90.28 | 225 min | `--time-steps 8` |
| **`snn_plif_t8`** | **62.73** | **92.07** | 227 min | `--time-steps 8 --lif-learnable` |
| `snn_mem2` | 57.96 | 89.37 | 133 min | `--readout membrane` |
| `snn_kwta` | 56.37 | 88.19 | 122 min | `--kwta-ratio 0.05` |
| `snn_dg4` | 57.41 | 88.89 | 170 min | `--kwta-ratio 0.05 --kwta-expand 4` |
| `snn_cal` | 57.26 | 88.48 | 148 min | `--calibrate` |
| `snn_r32`（1:1 回放） | 48.43 | 89.84 | 151 min | `--minibatch-size 32` |
| `snn_pshead`（固定正交头） | **61.37** | 90.22 | 106 min | `--fixed-head orthogonal` |
| `snn_all3`（三者全开） | 61.73 | 90.83 | ~241 min | `--fixed-head orthogonal --lif-learnable --time-steps 8` |

> `snn_all3` 是分两次跑完的（中途暂停 + `--resume`），所以 `results.json` 里的 `elapsed_s`
> 只统计了后半程；真实总时长约 241 分钟。暂停与续跑：

```bash
# 暂停：停掉进程即可（每个任务结束都会写 last.pt）
# 续跑：参数必须与原命令一致，tag 也要相同
python scripts/train.py ... --tag snn_all3 --resume runs/seq_cifar10_T8_w32_buf500_snn_all3
# 首行应输出 [resume] restored after task N; continuing at task N+1/5
```

基线命令：

```bash
python scripts/train.py --dataset seq_cifar10 --epochs 25 --buffer-size 500 \
    --batch-size 32 --width 32 --time-steps 4 --device cuda --tag snn_e25
```

最好的一组：

```bash
python scripts/train.py --dataset seq_cifar10 --epochs 25 --buffer-size 500 \
    --batch-size 32 --width 32 --time-steps 8 --lif-learnable --device cuda \
    --tag snn_plif_t8
```

> 运行时受显存/驱动影响较大（同一配置在不同时间可差 30%），上表的分钟数仅作参考。

## 回放比例扫描（reg_weight）

| run | 结果 | 开关 |
|---|---|---|
| `reg0.3_reg03` | 56.08 | `--reg-weight 0.3` |
| `reg075` | 55.78 | `--reg-weight 0.75` |
| `reg1.5_reg15` | 50.82 | `--reg-weight 1.5` |

（这三组是在更早的一版基线上跑的，绝对值不可与上表直接比较，只看相对关系。）

## 2 任务快验（**不可与 5 任务结果比较**）

以下运行的 `tasks_done = 2`，仅用于早期筛选。它们的绝对值系统性偏高，**这也是本项目
最重要的一条方法论教训**：任务数 ≤ 2 的消融会高估"保护旧任务"类改动——同一改动
（1:1 回放）在 2 任务上 +5.6、在 5 任务上在线性头下 −9.1（换类均值头后其实是 **+4.29**）。

| run | class-IL（2 任务） | 说明 |
|---|---|---|
| `replay32` | 77.55 | 1:1 回放，2 任务 |
| `proto_w00` ~ `proto4_C` | 68–72 | 特征原型约束六组 |
| `cal_la2` | 73.92 | LA 校准，2 任务 |

## 用类均值头重新评测（**重要**）

线性头在 class-IL 下的新类偏置随配置变化 **30 点以上**（README §7j / 报告 §8i），
所以任何结论都应该同时看两个头：

```bash
# 单个 checkpoint（几分钟，不训练）
python tools/eval_head_compare.py runs/seq_cifar10_T8_w32_buf500_snn_plif_t8

# 全部已完成的 5 任务运行（约 1 小时）
python tools/head_sweep.py            # 结果写入 report/head_sweep.json
```

`head_sweep.py` 会把 linear / NCM-cosine / NCM-euclidean 三种头的结果并排打出，
并按"最佳 NCM"重新排序。报告里所有主数字都取自这个文件。

## 报告与图表

```bash
python tools/collect_report_data.py    # 汇总 runs/*/results.json → report/results.json
python report/build_report.py          # 生成 report/*.docx
# 再导出 PDF（需要 LibreOffice）：
#   soffice --headless --convert-to pdf report/*.docx
```

## 已知的复现陷阱

1. **必须保留 `--tag`**，否则输出目录会变，`tools/collect_report_data.py` 找不到对应运行。
2. **`--proto-weight 0 --proto-margin-weight 0.3` 曾静默失效**（原型只在 `proto_weight > 0`
   时创建）。现在两者任一为正都会创建，并加了断言。
3. **`--sparse-layers none` 曾静默关闭 Eq.14**。现在会显式报错。
4. **单 seed**：绝对值 ±1–2 点在噪声内，只有跨多个检查点一致的趋势才值得解读。

## 用类均值头训练（--head-mode ncm）

把训练期的分类器从可学习线性层换成特征空间里的类均值（参数自由、每个 epoch 从缓冲区精确重建）：

`ash
python scripts/train.py --dataset seq_cifar10 --epochs 25 --buffer-size 500 \
    --batch-size 32 --width 32 --head-mode ncm --head-metric cosine \
    --device cuda --tag snn_ncm_train
`

要点：
- 每个模型副本（working / stable / plastic）各自持有一套原型，因为它们的特征空间不同；
- 未出现的类被掩到 −1e4（不贡献噪声）；
- 任务开始后**前两个 batch** 会额外重建一次（冷启动），否则新类没有原型、拿不到梯度。
