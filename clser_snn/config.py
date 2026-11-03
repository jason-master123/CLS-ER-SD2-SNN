"""Configuration for the CLS-ER + SD^2-SNN hybrid.

Three groups of knobs:

1. **Dataset / stream** — which continual benchmark and how it is split.
2. **CLS-ER scheme** — buffer, dual-memory EMA, consistency loss, fusion rule.
   Defaults follow the official implementation and the ICLR'22 tuned values for
   Seq-CIFAR-10 with a 500-sample buffer (`reg_weight 0.75`, both memories
   updated every step with alpha 0.999 / 0.99).
3. **SD^2-SNN substrate** — spiking backbone (T, width, LIF) and the optional
   SD^2 regularisers (decomposition, sparsity, sparse subspace).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class HybridConfig:
    # ------------------------------------------------------------- stream
    dataset: str = "seq_cifar10"        # seq_cifar10 | seq_cifar100 | seq_cifar100_20
    data_root: str = "./data"
    n_tasks: int = 5
    classes_per_task: int = 2
    num_classes: int = 0                # 0 -> derived from the stream at runtime
    batch_size: int = 32                # CLS-ER uses 32 for CIFAR
    num_workers: int = 0
    img_size: int = 32
    limit_train: int = 0                # >0 subsamples every task (debug)
    limit_test: int = 0

    # -------------------------------------------------- substrate / backbone
    backbone: str = "spiking"           # spiking | ann（ann = 稠密对照）
    width: int = 32                     # channels of the first stage
    time_steps: int = 4                 # T (SD^2-SNN Sec. 4.1 also reports T=8)
    tau: float = 2.0
    v_threshold: float = 1.0
    surrogate_alpha: float = 4.0
    detach_reset: bool = True
    lif_learnable: bool = False         # PLIF：每层可学习的膜时间常数（ICCV'21）
    kwta_ratio: float = 0.0             # 齿状回模式分离：保留 top-k% 特征（0 关闭）
    kwta_expand: int = 1                # 齿状回扩张倍数（>1 时先 C→r·C 再稀疏）
    fixed_head: str = "none"            # none | orthogonal（PS-SNN：固定正交类中心）
    # 注意：L2 归一化（余弦）在脉冲网上是退化的——放电率特征全部非负，而正交中心
    # 有正有负，cos 值被"均值方向"主导，各类 logits 几乎相同（实测 CE 卡在 log(10)）。
    # 因此默认关闭归一化：固定头直接作用在未归一化的码上。
    fixed_head_norm: bool = False
    fixed_head_scale: float = 1.0       # 余弦模式下才有意义（10~30），非归一化时保持 1
    readout: str = "rate"               # rate（时间平均放电率）| membrane（末层膜电位）
    readout_norm: bool = True           # 膜电位读出时加 LayerNorm
    amp: bool = False                   # bf16 autocast (fp16 unsafe for this net)

    # ------------------------------------------------ CLS-ER scheme (core)
    buffer_size: int = 500              # episodic memory (Table 1 of CLS-ER)
    minibatch_size: int = 10            # replay batch per step
    reg_weight: float = 0.75            # lambda of the consistency term, Eq.(1)
    stable_model_update_freq: float = 1.0
    stable_model_alpha: float = 0.999
    plastic_model_update_freq: float = 1.0
    plastic_model_alpha: float = 0.99
    # inference-time memory fusion: working | stable | plastic | mean | maxconf
    eval_mode: str = "working"

    # ------------------------------- SNN-native adaptations (see memories.py)
    # 1) full-state EMA: parameters + BN statistics + neuron thresholds
    ema_buffers: bool = True
    # 2) soft confidence gate instead of CLS-ER's hard per-exemplar argmax
    selector_mode: str = "soft"          # soft | hard
    selector_beta: float = 4.0
    selector_reliability: bool = True    # down-weight exemplars with too few spikes
    # 3) rate-domain consistency (scale-free) instead of logit MSE
    consistency_loss: str = "kl_rate"    # kl_rate | kl | mse
    consistency_tau: float = 2.0
    # 4) temporal distillation over the T timesteps (impossible in an ANN)
    temporal_weight: float = 0.0         # 0 disables
    temporal_tau: float = 2.0
    # 5) proximal consolidation towards the stable memory in parameter space
    proximal_weight: float = 0.0         # 0 disables
    # 6) replay coding: same direct encoding as the stream, or stochastic
    replay_encoding: str = "direct"      # direct | poisson
    # 7) homeostatic threshold calibration at the beginning of every task
    calibrate: bool = False
    rate_target: float = 0.05
    calibrate_batches: int = 8
    calibrate_rounds: int = 2
    # 8) stochastic evaluation: average over several independent encodings
    eval_passes: int = 1
    eval_encoding: str = "direct"        # direct | poisson

    # ------------------------------- logit calibration (see calibration.py)
    # comma separated list of: wa (weight aligning), la (logit adjustment),
    # bic (bias correction) | all | none
    calib_mode: str = "none"
    calib_la_mode: str = "mean"          # mean (CIL-relevant) | prior (long-tail LA)
    calib_tau: float = 1.0
    calib_augment: bool = True           # fit on augmented exemplars
    bic_steps: int = 150
    bic_lr: float = 0.01
    cosine_head: bool = False            # replace fc with a cosine classifier
    cosine_scale: float = 10.0

    # ------------------------------- inference-time classifier (see ncm.py)
    eval_head: str = "linear"            # linear | ncm
    ncm_temperature: float = 10.0
    ncm_metric: str = "cosine"           # cosine | euclidean

    # --------------------- feature-space prototype constraint (prototypes.py)
    proto_weight: float = 0.0            # 0 disables the training objective
    proto_temp: float = 0.1              # cosine-softmax temperature
    proto_min_samples: int = 4           # a class needs this many buffer items
    proto_margin_weight: float = 0.0     # inter-class hinge weight (0 = off)
    proto_margin: float = 0.1            # required cosine gap between classes
    # NOTE: prototypes are rebuilt exactly from the buffer every epoch
    # (``model.refresh_prototypes``); there is deliberately no EMA option, because
    # an EMA of normalised directions cancels out while the features rotate.

    # ------------------------------------ SD^2-SNN optional regularisers
    decompose: bool = False             # W ~= U V^T (Eq.16)
    rank_ratio: float = 0.75
    u_lr_ratio: float = 0.1             # eta_U = u_lr_ratio * eta_V (Eq.17)
    stability_weight: float = 0.0       # lambda of Eq.(18); 0 disables
    sparse_weight: float = 0.0          # lambda of Eq.(3) KL sparsity; 0 disables
    sparse_target: float = 0.05         # rho
    subspace_weight: float = 0.0        # lambda of Eq.(14); 0 disables
    subspace_eps: float = 0.02          # epsilon of Eq.(13)

    # ------------------------------------------------------------ training
    epochs: int = 50                    # per task (CLS-ER uses 50 for CIFAR)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"            # adamw | sgd
    lr_schedule: str = "cosine"         # cosine | constant
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "./runs"
    run_name: str = ""
    save_ckpt: bool = True
    log_every: int = 10                 # epochs between console lines
    stop_after_task: int = 0
    resume: str = ""                    # 从 runs/<tag> 的 last.pt 继续训练（跳过已完成的任务）

    def resolved(self) -> "HybridConfig":
        if getattr(self, "_resolved", False):
            return self
        from .datasets import STREAM_PRESETS
        preset = STREAM_PRESETS.get(self.dataset)
        if preset is not None:
            if self.n_tasks == 5 and self.classes_per_task == 2:   # untouched defaults
                self.n_tasks = preset["n_tasks"]
                self.classes_per_task = preset["classes_per_task"]
        self._resolved = True
        return self

    def to_dict(self):
        return asdict(self)


def add_config_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = HybridConfig().to_dict()
    for field, value in defaults.items():
        flag = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, dest=field, action="store_true", default=value)
            p.add_argument("--no-" + field.replace("_", "-"), dest=field,
                           action="store_false", default=argparse.SUPPRESS)
        elif isinstance(value, int):
            p.add_argument(flag, dest=field, type=int, default=value)
        elif isinstance(value, float):
            p.add_argument(flag, dest=field, type=float, default=value)
        else:
            p.add_argument(flag, dest=field, type=str, default=value)
    return p
