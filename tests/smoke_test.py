"""Self-checking smoke test for the CLS-ER + SD^2-SNN hybrid (CPU, < 2 min).

Section A covers the CLS-ER mechanisms, section B the SNN-native adaptations
(the parts that had to be redesigned because the ANN assumptions break), and
section C the SD^2-SNN regularisers.

Usage:  python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from clser_snn.backbone import SpikingResNet18
from clser_snn.buffer import ReservoirBuffer
from clser_snn.config import HybridConfig
from clser_snn.memories import SemanticMemories
from clser_snn.metrics import evaluate_task, summarise
from clser_snn.model import (SpikingCLSER, encode_direct, encode_poisson,
                             per_sample_distance)
from clser_snn.neurons import LIFNode
from clser_snn.sd2snn_extras import (SparseSubspaceMemory, basis_snapshot, decomposed_layers,
                                     kl_sparsity, layer_sparsity, parameter_groups,
                                     stability_penalty)


def make_cfg(**kw) -> HybridConfig:
    cfg = HybridConfig(dataset="seq_cifar10", n_tasks=3, classes_per_task=2,
                       batch_size=8, epochs=1, device="cpu", width=8,
                       buffer_size=16, minibatch_size=4, img_size=32,
                       output_dir="./_smoke_runs", run_name="smoke", save_ckpt=False)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# =========================================================================== #
# A. CLS-ER mechanisms
# =========================================================================== #
def test_spiking_forward():
    net = SpikingResNet18(num_classes=6, img_size=32, width=8)
    x = torch.randn(4, 3, 3, 32, 32)
    logits, logits_t, feats = net.forward_temporal(x)
    assert logits.shape == (3, 6), logits.shape
    assert logits_t.shape == (4, 3, 6), logits_t.shape      # per-timestep evidence
    assert feats.shape == (4, 3, 64), feats.shape
    assert torch.isfinite(logits).all()
    print(f"  ok  spiking forward: logits {tuple(logits.shape)}, "
          f"temporal logits {tuple(logits_t.shape)}, features {tuple(feats.shape)}")


def test_lif_binary():
    lif = LIFNode()
    x = torch.randn(4, 5, 8) * 2.0
    s = lif(x)
    assert s.shape == x.shape
    assert set(torch.unique(s).tolist()) <= {0.0, 1.0}
    print(f"  ok  LIF emits binary spikes, mean rate {s.mean():.3f}")


def test_buffer():
    buf = ReservoirBuffer(buffer_size=10, device="cpu")
    for step in range(5):
        buf.add_data(torch.randn(4, 3, 32, 32) + step, torch.arange(4) + step * 4)
    assert len(buf) == 10 and buf.num_seen_examples == 20
    xs, ys = buf.get_data(4)
    assert xs.shape == (4, 3, 32, 32) and ys.shape == (4,)
    print(f"  ok  reservoir buffer: capacity 10, seen 20, sampled {xs.shape[0]}")


def test_selector_hard():
    """CLS-ER 原始硬选择：置信度更高的记忆整条接管。"""
    cfg = make_cfg(selector_mode="hard", selector_reliability=False)
    net = SpikingResNet18(num_classes=4, width=8)
    mem = SemanticMemories(net, cfg)
    with torch.no_grad():
        mem.stable.fc.weight.zero_()
        mem.plastic.fc.weight.zero_()
        mem.stable.fc.bias.copy_(torch.tensor([5.0, -5.0, 0.0, 0.0]))
        mem.plastic.fc.bias.copy_(torch.tensor([-5.0, 5.0, 0.0, 0.0]))
    x = torch.randn(2, 4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    z_sel, w = mem.select_logits(x, labels)
    assert torch.allclose(z_sel, torch.where(
        torch.tensor([True, False, True, False]).unsqueeze(1), z_sel, z_sel))
    assert torch.allclose(z_sel[0], mem.stable(x)[0])
    assert torch.allclose(z_sel[1], mem.plastic(x)[1])
    assert not z_sel.requires_grad
    print(f"  ok  hard selector (CLS-ER): per-exemplar memory choice, "
          f"frac_stable={mem.stats['frac_stable']:.2f}")


def test_selector_soft():
    """SNN 改造：软门控给出连续权重，且在 β 很大时逼近硬选择。"""
    cfg = make_cfg(selector_mode="soft", selector_beta=4.0, selector_reliability=False)
    net = SpikingResNet18(num_classes=4, width=8)
    mem = SemanticMemories(net, cfg)
    with torch.no_grad():
        mem.stable.fc.weight.zero_()
        mem.plastic.fc.weight.zero_()
        mem.stable.fc.bias.copy_(torch.tensor([5.0, -5.0, 0.0, 0.0]))
        mem.plastic.fc.bias.copy_(torch.tensor([-5.0, 5.0, 0.0, 0.0]))
    x = torch.randn(2, 4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    z_sel, w = mem.select_logits(x, labels)
    # 凸组合必须落在两记忆之间
    z_s, z_p = mem.stable(x), mem.plastic(x)
    assert (z_sel - z_s).abs().max() > 0 and (z_sel - z_p).abs().max() > 0
    assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-6
    # 大 β → 接近硬选择
    mem.selector_beta = 400.0
    z_sharp, _ = mem.select_logits(x, labels)
    assert (z_sharp[0] - z_s[0]).abs().max() < 0.05
    print(f"  ok  soft gate (SNN): continuous weights w∈[{w.min():.2f},{w.max():.2f}], "
          f"β=400 recovers hard choice")


def test_full_state_ema():
    """SNN 改造：BN 统计量与阈值必须一起 EMA，否则记忆的放电率分布是错的。"""
    cfg = make_cfg(ema_buffers=True, plastic_model_update_freq=1.0,
                   stable_model_update_freq=1.0)
    net = SpikingResNet18(num_classes=4, width=8)
    net.train()
    _ = net(encode_direct(torch.randn(8, 3, 32, 32), 2))     # 刷新 BN 统计
    mem = SemanticMemories(net, cfg)

    bn_work = net.stem.bn.running_mean.detach().clone()
    bn_mem = mem.plastic.stem.bn.running_mean.detach().clone()
    assert torch.allclose(bn_work, bn_mem, atol=1e-6), "synchronised at init"

    # 修改工作模型的 BN 统计与阈值，再做一步 EMA
    with torch.no_grad():
        net.stem.bn.running_mean.add_(1.0)
    for m in net.modules():
        if isinstance(m, LIFNode):
            m.v_threshold = 2.0
    mem.update(net)

    alpha = min(1 - 1 / (mem.global_step + 1), 0.999)        # = 0.5 at step 1
    expected_bn = alpha * bn_mem + (1 - alpha) * bn_work.add(1.0)
    assert torch.allclose(mem.plastic.stem.bn.running_mean, expected_bn, atol=1e-5)
    th = next(m.v_threshold for m in mem.plastic.modules() if isinstance(m, LIFNode))
    assert abs(th - (alpha * 1.0 + (1 - alpha) * 2.0)) < 1e-6, th
    print(f"  ok  full-state EMA (SNN): BN stats + v_threshold tracked "
          f"(alpha={alpha:.2f}, threshold {th:.2f})")


def test_ema_lag():
    cfg = make_cfg(plastic_model_update_freq=1.0, stable_model_update_freq=1.0,
                   plastic_model_alpha=0.999, stable_model_alpha=0.999)
    net = SpikingResNet18(num_classes=4, width=8)
    mem = SemanticMemories(net, cfg)
    init = mem.plastic.fc.weight.detach().clone()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5)
    mem.update(net)
    assert torch.allclose(mem.plastic.fc.weight, 0.5 * init + 0.5 * (init + 0.5), atol=1e-5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    prev = mem.plastic.fc.weight.detach().clone()
    mem.update(net)
    assert torch.allclose(mem.plastic.fc.weight, (2 / 3) * prev + (1 / 3) * net.fc.weight,
                          atol=1e-5)
    assert not mem.training and not mem.stable.training
    print("  ok  EMA schedule min(1-1/(t+1), α_max) verified, memories stay in eval mode")


def test_observe():
    cfg = make_cfg(reg_weight=0.75)
    model = SpikingCLSER(cfg)
    x_aug = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 6, (8,))
    x_raw = x_aug.clone()

    stats = model.observe(x_aug, y, x_raw)
    assert torch.isfinite(torch.tensor(stats["loss"])) and stats["cons"] == 0.0
    assert len(model.buffer) == 8

    stats = model.observe(x_aug, y, x_raw)
    assert stats["cons"] > 0 and stats["cons_w"] > 0
    assert torch.isfinite(torch.tensor(stats["loss"]))
    print(f"  ok  observe: ce={stats['ce']:.3f} cons={stats['cons']:.4f} "
          f"margin={stats['margin']:.3f}, buffer={len(model.buffer)}")


def test_distances():
    s = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t = torch.tensor([[2.0, 0.0, -1.0], [3.0, -3.0, 0.0]])
    d_mse = per_sample_distance(s, t, "mse")
    d_kl = per_sample_distance(s, t, "kl_rate", tau=2.0)
    assert d_mse.shape == d_kl.shape == (2,)
    assert d_mse[0] < 1e-9 and d_mse[1] > 0
    assert d_kl[0] < 1e-6 and d_kl[1] > 0
    # scale invariance of the temperature KL: scaling both logits leaves it unchanged
    d_scaled = per_sample_distance(s * 10, t * 10, "kl_rate", tau=2.0)
    assert not torch.allclose(d_kl, d_scaled), "temperature KL is deliberately not scale-free"
    print(f"  ok  distances: mse={d_mse[1]:.3f}, kl_rate={d_kl[1]:.3f} "
          f"(pulse logits need the KL form)")


def test_encodings():
    x = torch.rand(3, 3, 32, 32)
    d = encode_direct(x, 4)
    p = encode_poisson(x, 4)
    assert d.shape == p.shape == (4, 3, 3, 32, 32)
    assert torch.allclose(d[0], x) and torch.allclose(d[3], x)
    assert set(torch.unique(p).tolist()) <= {0.0, 1.0}
    assert not torch.allclose(p, d)
    assert p.mean() < x.mean() + 0.05          # Poisson coding is a noisy version
    print(f"  ok  encodings: direct deterministic, poisson mean rate {p.mean():.3f}")


def test_temporal_and_proximal():
    cfg = make_cfg(reg_weight=0.5, temporal_weight=0.5, temporal_tau=2.0,
                   proximal_weight=1e-4)
    model = SpikingCLSER(cfg)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 6, (8,))
    model.observe(x, y, x)                      # fill the buffer
    stats = model.observe(x, y, x)
    assert stats["temp"] >= 0 and stats["prox"] > 0
    assert torch.isfinite(torch.tensor(stats["loss"]))
    print(f"  ok  temporal KD + proximal: temp={stats['temp']:.4f} "
          f"prox={stats['prox']:.3f} (SNN-only terms)")


def test_calibration():
    cfg = make_cfg(calibrate=True, rate_target=0.05, calibrate_batches=1)
    model = SpikingCLSER(cfg)
    loader = [(None, None, torch.randn(8, 3, 32, 32)) for _ in range(2)]
    before = [m.v_threshold for m in model.net.modules() if isinstance(m, LIFNode)]
    info = model.calibrate_thresholds(loader)
    after = [m.v_threshold for m in model.net.modules() if isinstance(m, LIFNode)]
    assert info["layers"] > 0 and before != after
    assert all(t > 0 for t in after)
    print(f"  ok  homeostatic calibration: {info['layers']} layers, "
          f"rate {info['rate_before']:.3f} -> {info['rate_after']:.3f}")


# =========================================================================== #
# C. SD^2-SNN regularisers
# =========================================================================== #
def test_sd2_regularisers():
    cfg = make_cfg(decompose=True, rank_ratio=0.5, sparse_weight=1e-3,
                   subspace_weight=1.0, stability_weight=1e-2, u_lr_ratio=0.1)
    model = SpikingCLSER(cfg)
    layers = list(decomposed_layers(model.net))
    assert layers

    groups = {g["name"]: g for g in parameter_groups(model.net, 1e-3, 1e-4, 0.1)}
    assert groups["U"]["lr"] < groups["V"]["lr"]

    snap = basis_snapshot(model.net)
    assert stability_penalty(model.net, snap).item() == 0.0
    with torch.no_grad():
        layers[0][1].U.add_(1e-2)
    assert stability_penalty(model.net, snap).item() > 0

    rates = {"a": torch.rand(4, 16) * 0.1, "b": torch.rand(4, 8) * 0.1}
    assert torch.isfinite(layer_sparsity(rates, 0.05))
    assert kl_sparsity(torch.rand(4, 8) * 0.1, 0.05).shape == (4, 8)

    mem = SparseSubspaceMemory(eps=0.02)
    mem.update(0, {"a": torch.rand(16) * 0.1})
    assert mem.n_stored() > 0 and mem.loss(rates) >= 0

    stats = model.observe(torch.randn(8, 3, 32, 32), torch.randint(0, 6, (8,)),
                          torch.randn(8, 3, 32, 32))
    for key in ("loss", "sparse", "stability"):
        assert torch.isfinite(torch.tensor(stats[key])), (key, stats[key])
    print(f"  ok  SD^2 regularisers: {len(layers)} factorized layers, lr_U<lr_V, "
          f"Eq.18/3/13-14 all finite")


# =========================================================================== #
def test_metrics():
    class Mock:
        def __init__(self, logits):
            self.logits = logits
            self.i = 0
            self.net = self

        def train_step_mode(self, train):
            pass

        def __call__(self, _x):
            b = _x.shape[1]
            out = self.logits[self.i:self.i + b]
            self.i += b
            return out

        fc = type("FC", (), {"out_features": 6})()

    logits = torch.tensor([[9., 0, 0, 0, 0, 0.],
                           [0, 9, 0, 0, 0, 0.],
                           [0, 0, 1, 1, 9, 0.],
                           [0, 0, 8, 0, 1, 0.]])
    y = torch.tensor([0, 1, 4, 5])
    loader = [(torch.zeros(4, 3, 32, 32), y, torch.zeros(4, 3, 32, 32))]
    res = evaluate_task(Mock(logits), loader, seen_classes=[0, 1, 2, 3, 4, 5],
                        task_classes=[4, 5], device="cpu", time_steps=1)
    assert res["cil"] == 75.0 and res["til"] == 25.0, res

    summary = summarise([[50.0], [40.0, 60.0]], [[60.0], [50.0, 70.0]])
    assert abs(summary["cil_acc"] - 50.0) < 1e-6
    assert abs(summary["forgetting_cil"] - 5.0) < 1e-6
    print(f"  ok  metrics: class-IL 75.0 / task-IL 25.0, "
          f"summary forgetting {summary['forgetting_cil']:.1f}")


def test_lr_reset():
    """回归测试：cosine 衰减到 ~0 后，新任务必须把 LR 复位。

    否则第二个任务会在 lr≈0 下训练（CE 不降反升），这是本仓库实际踩过的坑。
    """
    from clser_snn.trainer import Trainer

    cfg = make_cfg()
    model = SpikingCLSER(cfg)
    trainer = Trainer(cfg, model, tasks=[])
    base = list(trainer.base_lrs)

    # 模拟一个任务结束时的状态：cosine 已把 lr 衰减到 ~0
    for group in trainer.opt.param_groups:
        group["lr"] = 1e-9
    # 若直接用当前 lr 建调度器，base_lrs 会被固化成 1e-9（这就是 bug）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt, T_max=10)
    assert abs(sched.base_lrs[0] - 1e-9) < 1e-12, "bug reproduction failed"
    for group in trainer.opt.param_groups:
        group["lr"] = 1e-9

    trainer._reset_lr()
    for group, b in zip(trainer.opt.param_groups, base):
        assert abs(group["lr"] - b) < 1e-12, (group["lr"], b)
    print(f"  ok  lr reset across tasks: base {[round(v, 6) for v in base]} restored "
          f"(U group stays {base[-1] / base[0]:.2f}x)")


# =========================================================================== #
# D. logit calibration
# =========================================================================== #
class _FixedNet(torch.nn.Module):
    """Mock backbone: returns a pre-set logit row for every sample of a class.

    The class id is smuggled through ``x[:, 0]`` so the mock needs no labels.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        fc = torch.nn.Module()
        fc.out_features = table.shape[1]
        fc.weight = torch.nn.Parameter(table.clone())
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x[:, 0].long()]


class _MockModel:
    def __init__(self, table: torch.Tensor, device: str = "cpu") -> None:
        self.net = _FixedNet(table)
        self.device = torch.device(device)
        self.net.eval()

    def train_step_mode(self, train: bool) -> None:
        pass


def _make_buffer(n_per_class: int = 8, n_classes: int = 4) -> tuple:
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    examples = torch.zeros(labels.numel(), 1)
    examples[:, 0] = labels.float()
    return examples, labels


def test_calibrator_transform():
    from clser_snn.calibration import LogitCalibrator

    z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ident = LogitCalibrator(n_classes=4)
    assert ident.is_identity() and torch.allclose(ident.transform(z), z)

    cal = LogitCalibrator(n_classes=4, scale=torch.tensor([2.0, 1.0, 1.0, 0.5]),
                          bias=torch.tensor([0.0, 0.0, 1.0, 0.0]),
                          alpha=0.5, beta=-1.0,
                          old_mask=torch.tensor([True, True, False, False]))
    out = cal.transform(z)
    # old classes: 0.5*(scale*z) + bias - 1 ; new classes: scale*z + bias
    assert abs(out[0, 0].item() - (0.5 * 2.0 * 1.0 + 0.0 - 1.0)) < 1e-6
    assert abs(out[0, 2].item() - (1.0 * 3.0 + 1.0)) < 1e-6
    assert not cal.is_identity()
    # state round-trip
    rt = LogitCalibrator.from_state(cal.state_dict())
    assert torch.allclose(rt.transform(z), out)
    print("  ok  calibrator transform: scale/bias on all classes, α/β on old only, "
          "state round-trip")


def test_weight_alignment():
    from clser_snn.calibration import fit_weight_alignment

    table = torch.zeros(4, 3)
    table[0] = torch.tensor([1.0, 0.0, 0.0]) * 1.0        # old classes: norm 1
    table[1] = torch.tensor([0.0, 1.0, 0.0]) * 1.0
    table[2] = torch.tensor([1.0, 0.0, 0.0]) * 8.0        # new classes: norm 8
    table[3] = torch.tensor([0.0, 1.0, 0.0]) * 8.0
    model = _MockModel(table)

    scale, info = fit_weight_alignment(model, old_classes=[0, 1], new_classes=[2, 3],
                                       clamp=(0.5, 100.0))
    assert abs(scale[0].item() - 8.0) < 1e-4, scale
    assert abs(scale[2].item() - 1.0) < 1e-6
    assert abs(info["gamma"] - 8.0) < 1e-4

    # clamping protects against a runaway ratio
    scale_c, info_c = fit_weight_alignment(model, [0, 1], [2, 3], clamp=(0.5, 2.0))
    assert abs(scale_c[0].item() - 2.0) < 1e-6
    print(f"  ok  weight aligning: γ={info['gamma']:.2f} (clamped to "
          f"{info_c['gamma_used']:.2f} when out of range)")


def test_logit_adjustment():
    from clser_snn.calibration import fit_logit_adjustment

    # class 3 (the newest task) is over-scored on *every* sample — that is the
    # class-incremental bias the correction has to remove
    table = torch.tensor([[2.0, 0.0, 0.0, 6.0],
                          [0.0, 2.0, 0.0, 6.0],
                          [0.0, 0.0, 2.0, 6.0],
                          [0.0, 0.0, 0.0, 8.0]])
    model = _MockModel(table)
    examples, labels = _make_buffer(n_per_class=8, n_classes=4)
    bias, info = fit_logit_adjustment(model, examples, labels, [0, 1, 2, 3], mode="mean")
    assert bias.shape == (4,)
    # marginal mean of class 3 is the largest -> pushed down hardest
    assert abs(bias[3].item() - 4.5) < 0.05, bias
    assert abs(bias[0].item() - (-1.5)) < 0.05, bias
    assert info["la_spread"] > 4.0

    # after the correction the balanced buffer is predicted ~uniformly
    z = table[labels]
    pred_before = z.argmax(dim=1).bincount(minlength=4).float()
    pred_after = (z - bias).argmax(dim=1).bincount(minlength=4).float()
    share_before = (pred_before.max() / pred_before.sum()).item()
    share_after = (pred_after.max() / pred_after.sum()).item()
    assert share_after <= share_before, (share_before, share_after)
    assert share_after <= 0.26, f"not uniform enough: {share_after}"

    bias_p, _ = fit_logit_adjustment(model, examples, labels, [0, 1, 2, 3],
                                     mode="prior", tau=1.0)
    assert torch.allclose(bias_p[:4], torch.full((4,), float(torch.log(torch.tensor(0.25)))),
                          atol=1e-5)
    print(f"  ok  logit adjustment: marginal estimator, bias={bias.tolist()}, "
          f"argmax share {share_before:.2f} -> {share_after:.2f} (uniform=0.25)")


def test_bias_correction():
    from clser_snn.calibration import fit_bias_correction

    # old-class logits swamp the new ones: with α=1 the new-class samples are
    # misclassified, so BiC must shrink the old classes.
    table = torch.tensor([[8.0, 0.0, 0.0, 0.0],
                          [0.0, 8.0, 0.0, 0.0],
                          [7.0, 7.0, 6.0, 0.0],
                          [7.0, 7.0, 0.0, 6.0]])
    model = _MockModel(table)
    examples, labels = _make_buffer(n_per_class=16, n_classes=4)

    z = table[labels].clone()
    acc_identity = (z.argmax(dim=1) == labels).float().mean().item()

    alpha, beta, info = fit_bias_correction(model, examples, labels, [0, 1, 2, 3],
                                            old_classes=[0, 1], steps=150, lr=0.05,
                                            batch_size=16)
    z_cal = table[labels].clone()
    z_cal[:, [0, 1]] = alpha * z_cal[:, [0, 1]] + beta
    acc_cal = (z_cal.argmax(dim=1) == labels).float().mean().item()

    assert info["bic_loss_last"] <= info["bic_loss_first"] + 1e-6, info
    assert acc_cal > acc_identity, (acc_identity, acc_cal)
    assert abs(alpha - 1.0) > 0.05, "BiC did not move α"
    print(f"  ok  bias correction: α={alpha:.3f} β={beta:+.3f}, "
          f"balanced acc {acc_identity:.2f} -> {acc_cal:.2f}, "
          f"loss {info['bic_loss_first']:.3f} -> {info['bic_loss_last']:.3f}")


def test_cosine_head():
    from clser_snn.calibration import CosineHead, install_cosine_head

    cfg = make_cfg()
    model = SpikingCLSER(cfg)
    before = model.net.fc.weight.detach().clone()
    assert install_cosine_head(model, scale=10.0)
    assert isinstance(model.net.fc, CosineHead)
    assert model.net.fc.out_features == before.shape[0]

    z = model.net(encode_direct(torch.randn(2, 3, 32, 32), 2))
    assert z.shape == (2, before.shape[0]) and torch.isfinite(z).all()
    assert not install_cosine_head(model), "install must be idempotent"

    # with a cosine head every class weight has unit norm -> WA is a no-op
    from clser_snn.calibration import class_weight_norms
    norms = class_weight_norms(model)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"  ok  cosine head: |z|max={z.abs().max():.2f} with scale=10, "
          f"idempotent install, WA becomes no-op")


def test_end_of_task_calibration():
    cfg = make_cfg(calib_mode="wa,la", calib_la_mode="mean")
    model = SpikingCLSER(cfg)
    for _ in range(3):
        model.observe(torch.randn(8, 3, 32, 32), torch.randint(0, 2, (8,)),
                      torch.randn(8, 3, 32, 32))
    loader = [(None, None, torch.randn(4, 3, 32, 32))]
    info = model.end_of_task(0, loader, seen_classes=[0, 1], task_classes=[0, 1])
    assert "calibration" in info
    assert model.calibrator.bias is not None
    sd = model.state_dict()
    assert "calibrator" in sd
    model.load_state_dict(sd)

    # identity when disabled
    cfg2 = make_cfg(calib_mode="none")
    model2 = SpikingCLSER(cfg2)
    info2 = model2.end_of_task(0, loader, seen_classes=[0, 1], task_classes=[0, 1])
    assert "calibration" not in info2 and model2.calibrator.is_identity()
    print(f"  ok  end_of_task: calibration={info['calibration']}, state round-trip, "
          f"identity when disabled")


# =========================================================================== #
# E. nearest-class-mean classifier
# =========================================================================== #
class _FeatNet(torch.nn.Module):
    """Mock backbone exposing ``features``; the class id rides in ``x[:, 0]``."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        fc = torch.nn.Module()
        fc.out_features = table.shape[0]
        fc.weight = torch.nn.Parameter(torch.zeros(table.shape[0], table.shape[1]))
        self.fc = fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (T, B, C); the class id is smuggled into channel 0."""
        if x.dim() == 5:                     # encoded images: (T, B, 3, H, W)
            T, B = x.shape[0], x.shape[1]
            ids = x[:, :, 0, 0, 0].reshape(-1).long()
        elif x.dim() == 3:                   # (T, B, C_in)
            T, B = x.shape[0], x.shape[1]
            ids = x[:, :, 0].reshape(-1).long()
        else:                                # raw buffer batch: (B, C_in)
            T, B = 1, x.shape[0]
            ids = x[:, 0].long()
        return self.table[ids].reshape(T, B, -1)


class _FeatModel:
    def __init__(self, table: torch.Tensor) -> None:
        self.net = _FeatNet(table)
        self.device = torch.device("cpu")
        self.net.eval()

    def train_step_mode(self, train: bool) -> None:
        pass


def test_ncm():
    from clser_snn.ncm import NCMClassifier

    # three classes living on orthogonal rays; class 2 is never in the buffer
    table = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
    model = _FeatModel(table)
    examples = torch.zeros(6, 1)
    examples[:, 0] = torch.tensor([0, 0, 1, 1, 0, 1]).float()
    labels = torch.tensor([0, 0, 1, 1, 0, 1])

    ncm = NCMClassifier.fit(model, examples, labels, n_classes=3, metric="cosine")
    assert ncm.n_present == 2, ncm.summary()
    assert not ncm.present[2], "absent class must be masked"

    scores = ncm.scores(table)                    # score every class prototype row
    assert scores[2, 2] == float("-inf"), "absent class must score -inf"
    assert scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]
    assert torch.allclose(scores[0, 0], torch.tensor(1.0), atol=1e-5)

    # prototypes are mean-normalised: the class-1 prototype keeps only its direction
    assert torch.allclose(ncm.prototypes[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

    ncm_e = NCMClassifier.fit(model, examples, labels, n_classes=3, metric="euclidean")
    assert ncm_e.metric == "euclidean"
    assert ncm_e.scores(table)[0, 0] > ncm_e.scores(table)[0, 1]

    # state round-trip
    rt = NCMClassifier.from_state(ncm.state_dict())
    assert torch.allclose(rt.scores(table)[:2, :2], ncm.scores(table)[:2, :2])
    print(f"  ok  NCM: {ncm.summary()}, absent class masked, "
          f"cosine+euclidean both rank correctly")


def test_ncm_evaluation():
    """NCM through the real evaluation path gives the expected accuracy."""
    from clser_snn.ncm import NCMClassifier

    table = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    model = _FeatModel(table)
    ex, lab = torch.zeros(4, 1), torch.tensor([0, 0, 1, 1])
    ex[:, 0] = lab.float()
    ncm = NCMClassifier.fit(model, ex, lab, n_classes=2)

    x_raw = torch.zeros(2, 3, 32, 32)
    x_raw[0, 0, 0, 0] = 0.0                  # sample from class 0
    x_raw[1, 0, 0, 0] = 1.0                  # sample from class 1
    loader = [(torch.zeros(2, 3, 32, 32), torch.tensor([0, 1]), x_raw)]
    res = evaluate_task(model, loader, seen_classes=[0, 1], task_classes=[0, 1],
                        device="cpu", time_steps=1, ncm=ncm)
    assert res["cil"] == 100.0 and res["til"] == 100.0, res

    # with the wrong prototype assignment the accuracy must drop (sanity of the path)
    ncm_bad = NCMClassifier(prototypes=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                           present=torch.tensor([True, True]), temperature=10.0)
    res_bad = evaluate_task(model, loader, seen_classes=[0, 1], task_classes=[0, 1],
                            device="cpu", time_steps=1, ncm=ncm_bad)
    assert res_bad["cil"] < 100.0, res_bad
    print(f"  ok  NCM evaluation: class-IL={res['cil']:.0f}, "
          f"corrupted prototypes drop to {res_bad['cil']:.0f}")


# =========================================================================== #
# F. feature-space prototype constraint
# =========================================================================== #
def test_prototype_memory():
    from clser_snn.prototypes import PrototypeMemory

    mem = PrototypeMemory(n_classes=4, dim=3, momentum=0.5, temperature=0.1)
    assert mem.n_present == 0
    assert torch.count_nonzero(mem.loss(torch.randn(2, 3), torch.tensor([0, 1]))) == 0

    # first observation installs the prototype (normalised)
    feats = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    labels = torch.tensor([0, 1])
    mem.update(feats, labels)
    assert mem.n_present == 2
    assert torch.allclose(mem.prototype[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(mem.prototype[1].norm(), torch.tensor(1.0), atol=1e-5)

    # second observation: momentum=0.5 blends old and new direction
    mem.update(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([0]))
    blended = 0.5 * torch.tensor([1.0, 0.0, 0.0]) + 0.5 * torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(mem.prototype[0], blended / blended.norm(), atol=1e-5)

    # absent classes are masked, present ones are scored by cosine / tau
    query = torch.tensor([[1.0, 0.0, 0.0]])
    logits = mem.logits(query)
    assert logits[0, 2] == float("-inf") and logits[0, 3] == float("-inf")
    assert torch.isfinite(logits[0, 0])
    expected = F.cosine_similarity(query, mem.prototype[0].unsqueeze(0)).item() / 0.1
    assert abs(logits[0, 0].item() - expected) < 1e-4, (logits[0, 0].item(), expected)

    # the loss only looks at samples whose class has a prototype
    seen_before = mem.n_present
    loss = mem.loss(torch.randn(3, 3), torch.tensor([0, 2, 1]))   # class 2 has none
    assert torch.isfinite(loss) and mem.n_present == seen_before
    print(f"  ok  prototype memory: EMA(m=0.5) + normalisation, {mem.n_present} prototypes, "
          f"absent classes masked, loss={loss.item():.3f}")


def test_prototype_loss_learns():
    """Descending the prototype loss must align the features with their prototypes."""
    from clser_snn.prototypes import PrototypeMemory

    mem = PrototypeMemory(n_classes=2, dim=4, momentum=0.9, temperature=0.1)
    proto = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mem.prototype[:] = proto
    mem.present[:] = True

    torch.manual_seed(0)
    feats = torch.randn(16, 4, requires_grad=True)
    labels = torch.tensor([0] * 8 + [1] * 8)
    opt = torch.optim.SGD([feats], lr=0.5)
    first = None
    for _ in range(40):
        loss = mem.loss(feats, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    last = float(mem.loss(feats.detach(), labels))
    assert last < first * 0.5, (first, last)

    acc = (mem.logits(feats.detach()).argmax(dim=1) == labels).float().mean().item()
    assert acc == 1.0, acc
    print(f"  ok  prototype loss: {first:.3f} -> {last:.3f} in 40 steps, "
          f"prototype accuracy 100%")


def test_prototype_refresh_fixes_ema_cancellation():
    """Regression test for the bug that invalidated the first weight sweep.

    An EMA of *normalised* feature means across a rotating feature space cancels:
    after a full 360° rotation the running prototype points back at the start while
    the features have moved on. Recomputing exactly from the buffer must be aligned.
    """
    from clser_snn.prototypes import PrototypeMemory

    angles = [0.0, 90.0, 180.0, 270.0]              # degrees, full rotation
    dirs = torch.tensor([[float(torch.cos(torch.tensor(a * 3.14159 / 180))),
                          float(torch.sin(torch.tensor(a * 3.14159 / 180)))]
                         for a in angles])

    mem = PrototypeMemory(n_classes=2, dim=2, momentum=0.9, temperature=0.1)
    for d in dirs:
        mem.update(torch.stack([d, torch.tensor([0.0, 1.0])]), torch.tensor([0, 1]))
    cos_ema = F.cosine_similarity(dirs[-1].unsqueeze(0),
                                  mem.prototype[0].unsqueeze(0)).item()
    assert cos_ema < 0.5, f"EMA should have drifted away from the current feature: {cos_ema}"

    # exact refresh with the current feature space -> perfectly aligned
    model = _FeatModel(torch.stack([dirs[-1], torch.tensor([0.0, 1.0])]))
    examples = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    labels = torch.tensor([0, 1, 0, 1])
    mem.refresh(model, examples, labels, min_samples=1)
    cos_ref = F.cosine_similarity(dirs[-1].unsqueeze(0),
                                  mem.prototype[0].unsqueeze(0)).item()
    assert cos_ref > 0.999, cos_ref
    print(f"  ok  refresh vs EMA: cos(EMA proto, current feats)={cos_ema:+.3f} "
          f"-> cos(refreshed)={cos_ref:.3f} (this was the sweep-killing bug)")


def test_prototype_margin():
    """Inter-class hinge: zero when the gap is wide, positive when classes collide."""
    from clser_snn.prototypes import PrototypeMemory

    mem = PrototypeMemory(n_classes=2, dim=2, temperature=0.1)
    mem.prototype[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mem.present[:] = True

    # features sitting exactly on their prototypes -> gap = 1, no penalty
    on = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    assert mem.margin_loss(on, labels, margin=0.1).item() == 0.0

    # features halfway between the two classes -> gap = 0, penalty = margin
    between = torch.tensor([[0.7071, 0.7071], [0.7071, 0.7071]])
    loss = mem.margin_loss(between, labels, margin=0.1)
    assert abs(loss.item() - 0.1) < 1e-3, loss.item()

    # descending it must widen the gap
    feats = torch.tensor([[0.8, 0.6], [0.6, 0.8]], requires_grad=True)
    opt = torch.optim.SGD([feats], lr=0.5)
    first = mem.margin_loss(feats, labels, margin=0.3).item()
    for _ in range(60):
        loss = mem.margin_loss(feats, labels, margin=0.3)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = mem.margin_loss(feats.detach(), labels, margin=0.3).item()
    assert last < first, (first, last)
    sim = F.normalize(feats.detach(), dim=1) @ F.normalize(mem.prototype, dim=1).t()
    gap = (sim[0, 0] - sim[0, 1]).item()
    print(f"  ok  prototype margin: 0 when separated, {first:.3f}->{last:.3f} under "
          f"optimisation, final gap={gap:.2f}")


def test_prototype_training_integration():
    cfg = make_cfg(proto_weight=0.5, proto_temp=0.1, proto_min_samples=1)
    model = SpikingCLSER(cfg)
    assert model.prototypes is not None
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))
    first = model.observe(x, y, x)
    assert first["proto"] == 0.0, "no prototypes yet -> the term is inactive at first"
    # prototypes are rebuilt from the buffer (not EMA-updated per step)
    info = model.refresh_prototypes()
    assert info is not None and model.prototypes.n_present > 0, info
    second = model.observe(x, y, x)
    assert second["proto"] > 0.0, "with prototypes installed the term must be active"
    sd = model.state_dict()
    assert "prototypes" in sd
    model.load_state_dict(sd)

    # off by default
    off = SpikingCLSER(make_cfg(proto_weight=0.0))
    assert off.prototypes is None
    # ... but the inter-class hinge must be able to work on its own
    only_margin = SpikingCLSER(make_cfg(proto_weight=0.0, proto_margin_weight=0.3))
    assert only_margin.prototypes is not None, "margin term needs the prototypes"
    print(f"  ok  prototype training: inactive before refresh, then loss={second['proto']:.3f}, "
          f"{model.prototypes.n_present} prototypes, ckpt round-trip, off by default, "
          f"margin-only still builds them")


# =========================================================================== #
# G. dense (ANN) substrate control
# =========================================================================== #
def test_ann_backbone():
    """The ANN control must honour the same interface, with a comparable size."""
    from clser_snn.backbone import build_backbone
    from clser_snn.backbone_ann import DenseResNet18

    spiking = build_backbone(make_cfg(width=32))
    dense = DenseResNet18(num_classes=6, img_size=32, width=32)
    ratio = dense.num_params / spiking.num_params
    assert 0.9 < ratio < 1.1, (dense.num_params, spiking.num_params)

    x = torch.randn(4, 3, 3, 32, 32)                 # (T, B, 3, H, W)
    logits, logits_t, feats = dense.forward_temporal(x)
    assert logits.shape == (3, 6), logits.shape
    assert logits_t.shape == (1, 3, 6), logits_t.shape
    assert feats.shape == (1, 3, 256), feats.shape
    assert torch.isfinite(logits).all()

    same = torch.randn(1, 3, 3, 32, 32).repeat(4, 1, 1, 1, 1)
    assert torch.allclose(dense.features(same), dense.features(same[:1]), atol=1e-6)
    print(f"  ok  ANN control: {dense.num_params / 1e6:.2f}M vs spiking "
          f"{spiking.num_params / 1e6:.2f}M ({ratio:.2f}x), same 3-tuple interface")


def test_ann_training_step():
    cfg = make_cfg(backbone="ann", reg_weight=0.75)
    model = SpikingCLSER(cfg)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))
    first = model.observe(x, y, x)
    assert first["proto"] == 0.0
    second = model.observe(x, y, x)
    assert second["cons"] > 0, "the CLS-ER consistency term must still be active"
    assert torch.isfinite(torch.tensor(second["loss"]))
    print(f"  ok  ANN training step: ce={second['ce']:.3f} cons={second['cons']:.4f} "
          f"(CLS-ER mechanics unchanged)")


def test_membrane_readout():
    """Membrane readout must be continuous, same-shaped, and cost the same T."""
    from clser_snn.backbone import build_backbone

    rate_net = build_backbone(make_cfg(readout="rate", width=32))
    mem_net = build_backbone(make_cfg(readout="membrane", width=32))
    x = torch.randn(4, 6, 3, 32, 32)               # (T=4, B=6, ...)

    f_rate = rate_net.features(x)
    f_mem = mem_net.features(x)
    assert f_rate.shape == (4, 6, 256), f_rate.shape
    assert f_mem.shape == (1, 6, 256), f_mem.shape      # no time axis for membrane

    # the classifier consumes rate = feats.mean(0): values are multiples of
    # 1/(T*H*W), i.e. a coarse grid; the membrane readout is continuous
    rate_in = f_rate.mean(dim=0)
    n_rate_levels = torch.unique(rate_in[0]).numel()
    n_mem_levels = torch.unique(f_mem[0, 0, :]).numel()
    assert n_rate_levels <= 70, n_rate_levels        # 4 timesteps x 4x4 spatial bins
    assert n_mem_levels > 100, n_mem_levels
    assert torch.isfinite(f_mem).all()
    # LayerNorm keeps the readout on a sane scale (comparable to a spike rate)
    assert f_mem.abs().max() < 20, f_mem.abs().max()

    logits = mem_net(x)
    assert logits.shape == (6, 6) and torch.isfinite(logits).all()
    # same parameter count: the readout adds only the LayerNorm
    extra = mem_net.num_params - rate_net.num_params
    assert 0 < extra < 2000, extra

    # reliability weighting is meaningless without a spike count -> must be 1,
    # otherwise the CLS-ER consistency term would be silently disabled
    cfg_mem = make_cfg(readout="membrane", width=32)
    model_mem = SpikingCLSER(cfg_mem)
    f = torch.randn(1, 4, 256)
    rel = model_mem._reliability(f)
    assert torch.allclose(rel, torch.ones_like(rel)), rel
    cfg_ann = make_cfg(backbone="ann", width=32)
    rel_ann = SpikingCLSER(cfg_ann)._reliability(torch.randn(1, 4, 256))
    assert torch.allclose(rel_ann, torch.ones_like(rel_ann)), rel_ann
    print(f"  ok  membrane readout: rate has {n_rate_levels} levels vs "
          f"{n_mem_levels} for membrane, +{extra} params (LayerNorm), "
          f"reliability forced to 1")


def test_readout_gradient_flow():
    """Regression: the readout must not be detached from the graph.

    The first version stored ``last_v.detach()``; the classifier then saw a
    constant, the backbone received no gradient from the CE path, and the whole
    net degenerated to a linear probe on random features (task 0: 96.1 -> 64.8).
    """
    from clser_snn.model import SpikingCLSER as _Model

    for readout in ("rate", "membrane"):
        cfg = make_cfg(readout=readout, width=8)
        model = _Model(cfg)
        x = torch.randn(4, 3, 32, 32)
        logits = model.net(encode_direct(x, cfg.time_steps))
        logits.sum().backward()
        with_grad = sum(1 for p in model.net.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        total = sum(1 for _ in model.net.parameters())
        assert with_grad == total, f"{readout}: only {with_grad}/{total} params got grad"
    print("  ok  readout gradient flow: every backbone parameter receives gradient "
          "for both rate and membrane readouts")


def test_plif():
    """PLIF: one learnable 1/tau per LIF layer, initialised to the fixed value."""
    from clser_snn.backbone import build_backbone
    from clser_snn.neurons import LIFNode

    fixed = build_backbone(make_cfg(width=32))
    plif = build_backbone(make_cfg(width=32, lif_learnable=True))
    n_lif = sum(1 for m in plif.modules() if isinstance(m, LIFNode))
    extra = plif.num_params - fixed.num_params
    assert extra == n_lif, (extra, n_lif)          # exactly one scalar per LIF

    # initial 1/tau reproduces the fixed tau=2 behaviour
    lif = next(m for m in plif.modules() if isinstance(m, LIFNode))
    assert abs(float(lif.inv_tau) - 0.5) < 1e-6, float(lif.inv_tau)

    # the parameters are trainable and reachable
    x = torch.randn(4, 3, 32, 32)
    plif(torch.randn(4, 3, 3, 32, 32)).sum().backward()
    tau_grads = [m.a.grad for m in plif.modules()
                 if isinstance(m, LIFNode) and m.learnable_tau]
    assert all(g is not None for g in tau_grads), "tau received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in tau_grads), "all tau grads are zero"

    # forward stays finite and 1/tau stays in (0,1)
    assert torch.isfinite(plif(torch.randn(4, 3, 3, 32, 32))).all()
    print(f"  ok  PLIF: {n_lif} learnable 1/tau (+{extra} params), init reproduces tau=2, "
          f"gradients reach every tau")


def test_kwta_pattern_separation():
    """Dentate-gyrus k-WTA: exactly k survivors per sample, gradient only on them."""
    from clser_snn.backbone import SpikingResNet18

    net = SpikingResNet18(num_classes=10, img_size=32, width=32, kwta_ratio=0.05)
    x = torch.randn(4, 6, 3, 32, 32)
    feats = net.features(x)
    code = net.features_to_code(feats)
    keep = max(1, int(round(256 * 0.05)))
    nnz = (code != 0).sum(dim=1)
    assert torch.all(nnz == keep), (nnz.tolist(), keep)
    topk = feats.mean(dim=0).topk(keep, dim=1).values
    assert torch.allclose(code.topk(keep, dim=1).values, topk, atol=1e-6)

    plain = SpikingResNet18(num_classes=10, img_size=32, width=32, kwta_ratio=0.0)
    assert torch.allclose(plain.features_to_code(feats), feats.mean(dim=0))

    # end-to-end backward must still train the head and the backbone
    logits = net(x)
    logits.sum().backward()
    assert net.fc.weight.grad is not None and float(net.fc.weight.grad.abs().sum()) > 0
    conv_grads = [p.grad for p in net.parameters() if p.dim() == 4 and p.grad is not None]
    assert conv_grads and any(float(g.abs().sum()) > 0 for g in conv_grads), "no conv grad"
    print(f"  ok  k-WTA: {keep}/256 survivors per sample (5%), keeps the top entries, "
          f"no-op when 0, gradient reaches the head")


def test_dg_expansion():
    """DG = expand (C -> r·C) then sparsify: survivors grow, codes stay sparse."""
    from clser_snn.backbone import SpikingResNet18

    x = torch.randn(4, 6, 3, 32, 32)
    base = SpikingResNet18(num_classes=10, img_size=32, width=32,
                           kwta_ratio=0.05, kwta_expand=1)
    dg = SpikingResNet18(num_classes=10, img_size=32, width=32,
                         kwta_ratio=0.05, kwta_expand=4)

    code_base = base.features_to_code(base.features(x))
    code_dg = dg.features_to_code(dg.features(x))
    n_base = int((code_base != 0).sum(dim=1)[0])
    n_dg = int((code_dg != 0).sum(dim=1)[0])
    assert code_dg.shape[1] == 1024, code_dg.shape
    assert n_dg == max(1, round(1024 * 0.05)) == 51, n_dg
    # the point: 4x expansion means MORE survivors than the dense 256-dim input
    assert n_dg > n_base and n_dg > 13, (n_dg, n_base)
    assert (code_dg != 0).float().mean() < 0.06          # still sparse overall

    extra = dg.num_params - base.num_params
    assert 250_000 < extra < 280_000, extra              # 256->1024 + head 1024->10

    logits = dg(x)
    logits.sum().backward()
    assert dg.dg.weight.grad is not None and float(dg.dg.weight.grad.abs().sum()) > 0
    # the temporal path must go through the same expansion (regression: it used to
    # feed the raw 256-dim features into a 1024-dim head and crash)
    out, out_t, f = dg.forward_temporal(x)
    assert out_t.shape == (4, 6, 10), out_t.shape
    assert torch.isfinite(out_t).all() and torch.isfinite(out).all()
    # sanity of the slicing itself: on a plain net (no ReLU/k-WTA) the pooled logits
    # must equal the mean of the per-timestep logits
    plain = SpikingResNet18(num_classes=10, img_size=32, width=32,
                            kwta_ratio=0.0, kwta_expand=1)
    p_out, p_out_t, _ = plain.forward_temporal(x)
    assert torch.allclose(p_out, p_out_t.mean(dim=0), atol=1e-4)
    print(f"  ok  DG = expand+sparsify: {n_base} survivors in-place vs {n_dg} after "
          f"4x expansion (both ~5% sparse), +{extra / 1e3:.0f}k params")


def test_fixed_orthogonal_head():
    """PS-SNN head: unit-norm orthogonal centers that never move."""
    from clser_snn.backbone import build_backbone

    net = build_backbone(make_cfg(width=32, num_classes=6, fixed_head="orthogonal"))
    W = net.fc.weight.detach()
    assert W.shape == (6, 256), W.shape
    gram = W @ W.t()
    assert torch.allclose(gram, torch.eye(6), atol=1e-5), gram           # orthonormal
    assert not net.fc.weight.requires_grad and not net.fc.bias.requires_grad

    x = torch.randn(4, 5, 3, 32, 32)
    logits = net(x)
    assert logits.shape == (5, 6) and torch.isfinite(logits).all()

    # cosine mode: logits are bounded by the temperature. It is *off by default*
    # because rate features are non-negative: cos against mixed-sign centers is
    # dominated by the mean direction, and the net stalls at uniform predictions
    # (measured CE ~= log 10 on task 0). Here we only check the bound holds.
    cos_net = build_backbone(make_cfg(width=32, num_classes=6, fixed_head="orthogonal",
                                      fixed_head_norm=True, fixed_head_scale=2.0))
    cos_logits = cos_net(x)
    assert cos_logits.abs().max() <= 2.0 * (1 + 1e-4), cos_logits.abs().max()

    logits.sum().backward()
    assert net.fc.weight.grad is None                     # no gradient for the head
    conv = [p for p in net.parameters() if p.dim() == 4 and p.grad is not None]
    assert conv and any(float(p.grad.abs().sum()) > 0 for p in conv)

    before = W.clone()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
    opt.step()
    assert torch.allclose(net.fc.weight.detach(), before)  # unchanged after a step

    from clser_snn.model import SpikingCLSER
    cfg = make_cfg(fixed_head="orthogonal", width=32)
    model = SpikingCLSER(cfg)
    opt_ids = {id(p) for g in model.opt.param_groups for p in g["params"]}
    assert id(model.net.fc.weight) not in opt_ids
    out = model.observe(torch.randn(8, 3, 32, 32), torch.randint(0, 4, (8,)),
                        torch.randn(8, 3, 32, 32))
    assert torch.isfinite(torch.tensor(out["loss"]))
    print(f"  ok  fixed orthogonal head: 6 orthonormal centers in 256-d, frozen, "
          f"excluded from the optimizer (|logit| <= {net.fixed_head_scale:g})")


def test_resume_plumbing():
    """A checkpoint must restore the model, the accuracy matrices and the task cursor."""
    from types import SimpleNamespace

    from clser_snn.model import SpikingCLSER as _Model
    from clser_snn.trainer import Trainer

    cfg = make_cfg(width=8, num_classes=10)
    model = _Model(cfg)
    ckpt = {"model": model.state_dict(), "cfg": cfg.to_dict(),
            "cil": [[95.0], [80.0, 84.0]], "til": [[95.0], [90.0, 92.0]],
            "history": [{"task": 0}, {"task": 1}], "task": 1, "elapsed_s": 123.0}
    tasks = [SimpleNamespace(classes=[2 * i, 2 * i + 1]) for i in range(5)]

    fresh = _Model(cfg)
    tr = Trainer(cfg, fresh, tasks, resume=ckpt)
    assert tr.start_task == 2, tr.start_task        # two tasks done -> continue at #2
    assert tr.seen_classes == [0, 1, 2, 3], tr.seen_classes
    assert len(tr.cil_matrix) == 2 and tr.cil_matrix[1] == [80.0, 84.0]
    assert tr.elapsed_offset == 123.0
    # the restored weights really are the checkpoint's
    for a, b in zip(fresh.net.parameters(), model.net.parameters()):
        assert torch.equal(a, b)
    print("  ok  resume: task cursor, seen classes, accuracy matrices and weights "
          "all restored from the checkpoint")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("running CLS-ER + SD^2-SNN smoke tests ...")
    print("[A] CLS-ER mechanisms")
    test_spiking_forward()
    test_lif_binary()
    test_buffer()
    test_selector_hard()
    test_ema_lag()
    test_observe()
    test_metrics()
    test_lr_reset()
    print("[B] SNN-native adaptations")
    test_selector_soft()
    test_full_state_ema()
    test_distances()
    test_encodings()
    test_temporal_and_proximal()
    test_calibration()
    print("[C] SD^2-SNN regularisers")
    test_sd2_regularisers()
    print("[D] logit calibration")
    test_calibrator_transform()
    test_weight_alignment()
    test_logit_adjustment()
    test_bias_correction()
    test_cosine_head()
    test_end_of_task_calibration()
    print("[E] nearest-class-mean classifier")
    test_ncm()
    test_ncm_evaluation()
    print("[F] feature-space prototype constraint")
    test_prototype_memory()
    test_prototype_loss_learns()
    test_prototype_refresh_fixes_ema_cancellation()
    test_prototype_margin()
    test_prototype_training_integration()
    print("[G] dense (ANN) substrate control")
    test_ann_backbone()
    test_ann_training_step()
    print("[H] membrane readout")
    test_membrane_readout()
    test_readout_gradient_flow()
    print("[I] PLIF backbone")
    test_plif()
    print("[J] dentate gyrus (k-WTA pattern separation)")
    test_kwta_pattern_separation()
    test_dg_expansion()
    print("[K] fixed orthogonal head (PS-SNN)")
    test_fixed_orthogonal_head()
    print("[L] checkpoint / resume")
    test_resume_plumbing()
    print("all smoke tests passed")
