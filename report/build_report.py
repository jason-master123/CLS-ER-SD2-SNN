"""Build the CLS-ER + SD^2-SNN report as a DOCX (exported to PDF afterwards)."""

from __future__ import annotations

import json
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
CN = "Microsoft YaHei"
MONO = "Consolas"


# --------------------------------------------------------------------------- #
def set_font(run, size=None, bold=None, color=None, mono=False):
    name = MONO if mono else CN
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name if mono else CN)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def _emit_runs(p, text, size, bold=False, color=None):
    """Emit a paragraph's runs, honouring inline ``**bold**`` and backtick code."""
    for i, part in enumerate(str(text).split("`")):
        if not part:
            continue
        if i % 2 == 1:                     # inside backticks -> monospace
            set_font(p.add_run(part), size=size - 0.5, bold=bold, color=color, mono=True)
            continue
        for j, sub in enumerate(part.split("**")):
            if not sub:
                continue
            set_font(p.add_run(sub), size=size,
                     bold=(bold or (j % 2 == 1)), color=color)


def para(doc, text="", size=10.5, bold=False, space_after=6, align=None,
         color=None, indent_first=False, keep_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.35
    pf.keep_with_next = keep_next
    if indent_first:
        pf.first_line_indent = Pt(21)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if text:
        _emit_runs(p, text, size, bold, color)
    return p


def heading(doc, text, level=1):
    sizes = {1: 15, 2: 12.5}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    run = p.add_run(text)
    set_font(run, size=sizes.get(level, 11), bold=True,
             color=(0x1F, 0x3B, 0x63) if level == 1 else (0x2E, 0x52, 0x7A))
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(16)
    pf.first_line_indent = Pt(-11)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.3
    set_font(p.add_run("• "), size=size)
    _emit_runs(p, text, size)
    return p


def code_block(doc, lines, size=8.5):
    for line in lines:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(0)
        pf.space_before = Pt(0)
        pf.left_indent = Pt(10)
        pf.line_spacing = 1.0
        run = p.add_run(line if line else " ")
        set_font(run, size=size, mono=True, color=(0x33, 0x33, 0x33))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _repeat_header(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def _cant_split(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    trpr.append(OxmlElement("w:cantSplit"))


def table(doc, headers, rows, widths=None, size=9, caption_text=None,
          highlight_rows=(), numeric_from=1):
    if caption_text:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(8)
        cap.paragraph_format.space_after = Pt(3)
        cap.paragraph_format.keep_with_next = True
        set_font(cap.add_run(caption_text), size=9, bold=True, color=(0x33, 0x33, 0x33))

    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        set_font(p.add_run(str(h)), size=size, bold=True)
    _repeat_header(t.rows[0])

    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        _cant_split(t.rows[-1])
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from
                           else WD_ALIGN_PARAGRAPH.LEFT)
            set_font(p.add_run(str(v).replace("**", "")), size=size,
                     bold=(r_i in highlight_rows))

    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def fmt(v, nd=2):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- #
def build() -> str:
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        blob = json.load(f)
    runs = blob["runs"]

    def r(tag):
        return runs[tag]

    snn = r("snn_e25")["final"]
    ann = r("ann_e25")["final"]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.9)
    style = doc.styles["Normal"]
    style.font.name = CN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN)

    # ------------------------------------------------------------- cover
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(64)
    set_font(p.add_run("CLS-ER + SD²-SNN"), size=26, bold=True, color=(0x1F, 0x3B, 0x63))

    for text, size, color in [
        ("把互补学习系统的快慢机制移植到脉冲神经网络", 13, (0x33, 0x33, 0x33)),
        ("六处改造 · 五条否定路线 · 一份可复现的实测记录", 11, (0x55, 0x55, 0x55)),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(text), size=size, color=color)

    doc.add_paragraph().paragraph_format.space_after = Pt(34)

    table(doc, ["项目", "内容"], [
        ["组合的两条论文线", "CLS-ER（ICLR'22，互补学习系统 + 双语义记忆 + 回放）\n"
                              "SD²-SNN（脉冲骨干、直接编码 T 步、可选结构化分解）"],
        ["代码", "CLS-ER+SD2-SNN（PyTorch 2.14 + CUDA 12.6，10 个模块，30 项自检）"],
        ["基准", "Seq-CIFAR-10（5 任务 × 2 类），buffer 500，Class-IL / Task-IL"],
        ["平台", "RTX 4060 Laptop 8 GB / Windows / bf16"],
        ["实验量", "21 次训练运行，约 14 小时 GPU"],
        ["日期", "2026-10-30 ~ 2026-10-31"],
    ], widths=[3.0, 13.9], size=9.5, numeric_from=99)

    doc.add_page_break()

    # ------------------------------------------------------------- summary
    heading(doc, "1  摘要")
    para(doc, "本项目把 CLS-ER 的互补学习系统（双语义记忆 + 选择性一致性约束 + 回放）移植到 "
              "SD²-SNN 的脉冲骨干上，并系统回答了三个问题：哪些 ANN 机制在脉冲网络里会失效、"
              "改造后能拿回多少、以及剩下的差距在哪里。", indent_first=True)

    bullet(doc, "**六处机制必须改造**：整状态 EMA（参数 + BN 统计 + 阈值）、软门控选择器、"
                "率域温度 KL、时序蒸馏、稳态阈值校准、近端约束。直接照搬 ANN 口径只有 39.50。")
    bullet(doc, f"**改造后 class-IL = {snn['cil_acc']:.2f}**（task-IL {snn['til_acc']:.2f}），"
                f"相对 ANN 口径 **+{snn['cil_acc'] - ann['cil_acc']:.1f} 点**。")
    bullet(doc, f"**进一步把基底改对（PLIF + T=8）后达到 {r('snn_plif_t8')['final']['cil_acc']:.2f}**"
                f"（task-IL {r('snn_plif_t8')['final']['til_acc']:.2f}，F_cil "
                f"{r('snn_plif_t8')['final']['forgetting_cil']:.2f}），比基线高 5.22 点，"
                f"与稠密 ANN 的差距从 8.05 缩到 **2.83**。")
    bullet(doc, "**八条“减少遗忘”的路线全部为零或负**：logit 校准（三变体）、NCM 换头（+2.1，边际）、"
                "特征原型约束（六组）、1:1 回放（−9.1）、一致性权重上调（−5.0）、"
                "稳态阈值校准（−0.25）、T=8（+1.29 但 2 倍算力）、膜电位读出（+0.45）。")
    bullet(doc, "**两个“更根本”的假设也被证伪**：读出量化（连续膜电位 ≈ 9 档量化）、"
                "表征正交性（齿状回 k-WTA 扩张到 51 个稀疏单元后 class-IL 仍持平）。")
    bullet(doc, "**差距已可拆分**：稠密骨干对照（同参数同预算）拿到 65.56，把 17.7 点拆成"
                "**9.66（预算/配方）+ 8.05（脉冲基底）**。基底差距落在每个任务的绝对精度上，"
                "而不是遗忘——这推翻了此前八条“减少遗忘”路线的共同前提。")
    bullet(doc, "**一条方法论教训**：任务数 ≤ 2 的消融会系统性高估“保护旧任务”类改动——"
                "同一改动在 2 任务上 +5.6、在 5 任务上 −9.1。")
    bullet(doc, "**与论文的差距**：我们的稠密对照也只有 65.56（论文 75.22），"
                "所以那 9.66 点与脉冲无关，来自 25 vs 50 epochs 和简化掉的配方。")

    # ------------------------------------------------------------- goal
    heading(doc, "2  目标与两条论文线的交接面")
    para(doc, "两个方法负责的事情是正交的，这也是移植能成立的先决条件：", indent_first=True)
    table(doc, ["层", "CLS-ER 提供", "SD²-SNN 提供"], [
        ["学什么", "working + stable/plastic 双 EMA 记忆\n选择器式一致性约束（Eq.1）\nreservoir 回放",
         "—"],
        ["用什么学", "—", "LIF + ATan 替代梯度\nSpiking-ResNet-18，直接编码 T 步\n时间平均放电率读出"],
        ["可选增强", "—", "W≈UVᵀ 分解 / KL 稀疏 / 稀疏子空间（默认关闭）"],
    ], widths=[2.4, 7.2, 7.2], size=9, caption_text="表 1  两条论文线的分工", numeric_from=99)

    para(doc, "于是移植的接口只有三处：输入编码、特征/放电率读出、以及“什么算一次前向”。"
              "CLS-ER 的三个模型（working / stable / plastic）共享同一套骨干结构，记忆只是权重副本。",
         indent_first=True)

    # ------------------------------------------------------------- six fixes
    heading(doc, "3  六处必须改造的机制")
    para(doc, "原版 CLS-ER 的每条机制都隐含 ANN 假设，在脉冲网络里逐条失效。"
              "下表给出假设、现实、替换与证据：", indent_first=True)
    table(doc, ["#", "原机制", "ANN 隐含假设 → SNN 现实", "替换为", "证据"], [
        ["①", "只 EMA 参数 W",
         "函数只由 W 决定 → 函数由 (W, BN 统计, v_th, τ) 共同决定，BN 统计直接决定放电率尺度",
         "整状态 EMA（参数 + buffers + 阈值）",
         "ANN 口径下 frac_stable 前 4 任务恒 0.000"],
        ["②", "按真值置信度硬选记忆",
         "softmax 置信度可靠 → T 步平均放电率方差 ~ r(1−r)/T",
         "软门控 w=σ(βΔc)，β→∞ 退化回硬选择",
         "低置信度区硬选择被噪声主导"],
        ["③", "logits 上的 MSE",
         "logits 连续、量级稳定 → 脉冲 logits 是放电率的线性映射，量纲小且随 T 漂移",
         "率域温度 KL + 脉冲数可靠性加权",
         "同尺度差下 MSE 爆到 8.3、KL 有界"],
        ["④", "单一时刻输出",
         "只有一个监督点 → T 个时刻各自产生证据，“何时发放”是信息",
         "时序蒸馏（逐 timestep logits 的 KL）",
         "ANN 无此维度，故为新增项"],
        ["⑤", "无活动调节",
         "激活不会死/饱和 → 放电率会崩塌或爆炸（SNN 特有失败模式）",
         "稳态阈值校准（homeostatic）",
         "单任务 30.40 → 42.00（+11.6）"],
        ["⑥", "仅 logits 蒸馏",
         "梯度无偏 → 代理梯度有偏，参数沿噪声方向漂移",
         "近端约束 λ‖θ_W−θ_stable‖²",
         "可选，默认关闭"],
    ], widths=[0.7, 3.4, 5.6, 3.6, 3.6], size=8, numeric_from=99,
        caption_text="表 2  六处改造（①–③ 是主实验捆绑的三项，④–⑥ 为可选）")

    para(doc, "两条最硬的证据：", space_after=3, keep_next=True)
    bullet(doc, "**放电率**：未校准的网络在任务 0 开始几乎不发放（全网平均 0.001），"
                "校准后 0.065；学完一个任务又涨到 0.385，下个任务开始前再校准回 0.089。"
                "ANN 机制在这种情况下根本没有可用信号。")
    bullet(doc, "**frac_stable**：ANN 口径下前 4 个任务恒为 0.000——stable 记忆一次都没被选中过，"
                "因为它的 BN 统计冻结在初始化副本上。整状态 EMA 之后它全程在 0.3–0.9 起伏，"
                "任务边界处升高（慢记忆接管）。")

    # ------------------------------------------------------------- main result
    heading(doc, "4  主实验：SNN 原生适配 vs ANN 口径")
    para(doc, "同 seed、同预算（width 32、T=4、buffer 500、batch 32、25 epochs × 5 任务），"
              "唯一差别是上表 ①②③ 的机制口径：", indent_first=True)
    table(doc, ["配置", "class-IL", "task-IL", "F_cil", "F_til", "BWT", "耗时"], [
        ["**SNN 原生**", f"{snn['cil_acc']:.2f}", f"{snn['til_acc']:.2f}",
         f"{snn['forgetting_cil']:.2f}", f"{snn['forgetting_til']:.2f}",
         f"{snn['bwt']:.2f}", f"{snn['elapsed_s'] / 60:.0f} min"],
        ["ANN 口径", f"{ann['cil_acc']:.2f}", f"{ann['til_acc']:.2f}",
         f"{ann['forgetting_cil']:.2f}", f"{ann['forgetting_til']:.2f}",
         f"{ann['bwt']:.2f}", f"{ann['elapsed_s'] / 60:.0f} min"],
        ["差", f"+{snn['cil_acc'] - ann['cil_acc']:.2f}",
         f"+{snn['til_acc'] - ann['til_acc']:.2f}",
         f"{snn['forgetting_cil'] - ann['forgetting_cil']:.2f}",
         f"{snn['forgetting_til'] - ann['forgetting_til']:.2f}",
         f"{snn['bwt'] - ann['bwt']:.2f}", "—"],
    ], widths=[2.8, 2.4, 2.3, 2.1, 2.1, 2.2, 2.0], size=9, highlight_rows=(0,),
        caption_text="表 3  五任务主结果（Class-IL 为 5 个任务的平均）")

    def matrix_rows(tag):
        m = r(tag)["cil_matrix"]
        rows = []
        for j, row in enumerate(m):
            cells = [fmt(v, 1) if isinstance(v, float) else "" for v in row]
            rows.append([f"t{j}"] + cells + [""] * (5 - len(cells)))
        return rows

    table(doc, ["学完后", "t0", "t1", "t2", "t3", "t4"], matrix_rows("snn_e25"),
          widths=[2.0, 2.9, 2.9, 2.9, 2.9, 2.9], size=8.5,
          caption_text="表 4  SNN 原生的 Class-IL 矩阵（行 = 学完第 j 个任务，列 = 任务 i）")
    table(doc, ["学完后", "t0", "t1", "t2", "t3", "t4"], matrix_rows("ann_e25"),
          widths=[2.0, 2.9, 2.9, 2.9, 2.9, 2.9], size=8.5,
          caption_text="表 5  ANN 口径的 Class-IL 矩阵")

    para(doc, "三点分析：", space_after=3, keep_next=True)
    bullet(doc, "**单任务上限完全相同**（96.1 vs 96.2）——骨干能力一致，差异纯粹来自记忆机制。")
    bullet(doc, "**差距集中在“中间任务被挤死”**：ANN 口径下 t1/t2 在学完 4 个任务后只剩 7.3 / 5.4"
                "（4 分类随机为 25%），SNN 口径保住 22.7 / 31.5；最老任务也更好（72.8 vs 62.7）。")
    bullet(doc, "**task-IL 也领先 8.6 点**（89.0 vs 80.4），说明改善不只是重排 logits——"
                "整状态 EMA 让记忆真的成了一个行为自洽的慢速教师。")

    para(doc, "必须说清的局限：这一组比较的是 ①②③ 的**捆绑**，没有逐项拆开。"
              "“总共值 +18 点”是可靠的，“每项各值多少”还不知道；且只跑了单个 seed。",
         indent_first=True)

    # ------------------------------------------------------------- negatives
    doc.add_page_break()
    heading(doc, "5  五条后续路线，全部为负")
    para(doc, "主实验之后，围绕“task-IL 89 而 class-IL 57.5 之间那 31 点”继续追，"
              "试了五条路线。每条都有明确的机制假设，每条都被实测否定。", indent_first=True)

    table(doc, ["路线", "机制假设", "实测", "判定"], [
        ["logit 校准\n(WA/LA/BiC/余弦头)", "瓶颈是跨任务 logit 偏置，事后修正即可",
         "三变体 −2.8 / −4.4 / −5.1", "否定"],
        ["NCM 换头\n(最近类均值)", "线性头无法跨任务比较，换掉它",
         "57.51 → 59.64（+2.1）", "边际"],
        ["特征原型约束\n(吸引 + 类间铰链)", "特征缺跨任务结构，训练时加约束",
         "六组全负（2 任务快验 −0.6 ~ −2.5）", "否定"],
        ["1:1 回放\n(minibatch 10→32)", "旧类只拿到 24% 梯度预算",
         "2 任务 +5.6 → **5 任务 −9.1**", "否定"],
        ["一致性权重\n(0.75→1.5)", "主损失权重是照搬 ANN 的",
         "−5.0（0.3 与 0.75 打平）", "否定"],
    ], widths=[3.6, 5.0, 5.0, 1.8], size=8.5, numeric_from=99,
        caption_text="表 6  五条路线的假设、实测与判定")

    heading(doc, "5.1  logit 校准：两次假设被证伪", level=2)
    para(doc, "实现了四个组件（权重对齐 WA、logit 调整 LA、偏置修正 BiC、余弦分类头），"
              "全部是 z → a⊙z + b 形式的后处理变换，不改网络、不影响记忆。", indent_first=True)
    table(doc, ["变体", "t1 class-IL", "判定", "说明"], [
        ["基线（无校准）", "79.00", "—", "—"],
        ["WA+LA，统计量在增广样本上估", "76.25", "−2.75", "假设：增广导致统计偏差"],
        ["WA+LA，统计量在干净样本上估", "74.60", "−4.40", "**证伪**：换成干净样本更差"],
        ["LA，边缘均值估计器", "73.92", "−5.08", "**证伪**：修正估计量后更差"],
    ], widths=[6.4, 2.6, 2.2, 5.0], size=8.5,
        caption_text="表 7  三个校准变体（2 任务、8 epochs 口径）")

    para(doc, "两次证伪指向同一件事：问题不在“统计量估在哪儿”也不在“估计量的形式”，"
              "而在事后校正这条路本身。机理是脉冲 logits 活在放电率尺度上，"
              "该尺度随任务增加持续漂移（BN 统计、阈值都在变），静态的逐类偏置追不上这种漂移。",
         indent_first=True)

    heading(doc, "5.2  NCM：+2.1 点，但推翻了我的诊断", level=2)
    para(doc, "在同一个 checkpoint 上（完全相同的权重与 buffer）只换决策规则：", indent_first=True)
    table(doc, ["决策规则", "class-IL", "task-IL", "每任务 Class-IL"], [
        ["线性头", "57.51", "88.98", "[40.1, 31.8, 51.9, 69.8, 94.0]"],
        ["NCM（余弦）", "59.64", "88.64", "[47.5, 33.2, 51.4, 74.2, 91.8]"],
        ["NCM（欧氏）", "59.64", "87.95", "[49.3, 35.7, 51.8, 69.6, 91.8]"],
    ], widths=[3.2, 2.4, 2.4, 8.2], size=8.5,
        caption_text="表 8  同权重换头的免训练对比")

    para(doc, "关键在第二列的差：如果瓶颈真在分类头，NCM 应该接近 task-IL 的 89，实际只有 59.6。"
              "**task-IL 89 是虚高的**——在 2 个类里做二分类，几乎任何特征空间都能做到 85+，"
              "它并不说明特征具备跨任务可比性。这条结果把诊断从“头坏了”改成“特征空间缺跨任务结构”。",
         indent_first=True)

    heading(doc, "5.3  特征原型约束：六组全负，且是同一个模式", level=2)
    table(doc, ["配置", "t0", "t1 class-IL", "旧任务", "新任务", "F_cil"], [
        ["基线", "91.5", "72.00", "67.4", "76.6", "12.07"],
        ["吸引 λ=0.1", "94.2", "71.00", "63.4", "78.7", "15.40"],
        ["吸引 λ=0.3", "91.7", "68.17", "60.4", "76.0", "15.68"],
        ["吸引 λ=1.0", "92.7", "70.00", "60.1", "79.9", "16.28"],
        ["吸引 0.05 + 铰链 0.1", "90.7", "70.45", "65.2", "75.8", "12.77"],
        ["铰链 0.5 / λ=1.0", "93.7", "71.42", "65.5", "77.4", "14.10"],
    ], widths=[5.0, 2.0, 2.6, 2.2, 2.2, 2.2], size=8.5,
        caption_text="表 9  原型约束六组（2 任务、8 epochs，与基线同预算）")

    para(doc, "三条稳定规律：约束本身**收敛**（原型损失 0.7–0.8，远低于随机 1.39；"
              "铰链能从 0.076 涨到 0.384，说明确实咬住了）→ 当前任务**一致变好** → "
              "旧任务**一致变差**。也就是说它不是“实现没生效”，而是“生效了但方向不对”："
              "类内吸引只把每个簇各自收紧，不约束簇与簇的相对布局，而后者才是 CIL 的瓶颈。",
         indent_first=True)

    heading(doc, "5.4  1:1 回放：2 任务的灵丹，5 任务的毒药", level=2)
    para(doc, "这是本项目最重要的一课。`--minibatch-size` 从项目开始一直对齐 CLS-ER 官方默认 10"
              "（每步回放 10 个旧样本，旧类只占 24% 的梯度预算）。", indent_first=True)
    table(doc, ["预算", "t0", "t1", "t2", "t3", "t4", "class-IL", "F_cil"], [
        ["24% 回放（基线）", "96.1", "79.0", "64.1", "55.2", "57.5", "57.51", "33.49"],
        ["1:1 回放", "97.2", "81.0", "60.8", "52.1", "48.4", "**48.43**", "**43.31**"],
    ], widths=[3.4, 1.7, 1.7, 1.7, 1.7, 1.7, 2.4, 2.0], size=8.5,
        caption_text="表 10  1:1 回放的逐检查点结果（5 任务 × 25 epochs）")

    para(doc, "但在 2 任务、8 epochs 的快验里，同一改动是 **+5.6**（72.0 → 77.6，旧任务 67.4 → 77.0、"
              "F_cil 12.07 → 8.38），看起来像重大突破。", indent_first=True)
    para(doc, "**为什么 2 任务会骗人**：只有 1 个旧任务时，32 个回放样本能均匀覆盖它（每类约 16 个），"
              "所以“提高回放权重”必然有效。任务数一多，同样的预算被摊到 4 个旧任务上，"
              "**每个旧类每步只剩 2–3 个样本**——信号不是变强而是变碎。证据是中间任务受害最重"
              "（t1 22.7 → 16.2、t2 31.5 → 23.2），而最老与最新的任务反而都略有提升。",
         indent_first=True)

    heading(doc, "5.5  一致性权重：0.75 是对的，敏感性是单向的", level=2)
    table(doc, ["λ", "t0", "t1", "t2（3 任务）", "cons 占总损失", "判定"], [
        ["0.3", "92.7", "73.9", "56.06", "5–10%", "与 0.75 打平"],
        ["0.75（默认）", "92.7", "71.9", "55.78", "6–13%", "基准"],
        ["1.5", "89.9", "66.7", "50.82", "—", "明显更差"],
    ], widths=[2.6, 1.9, 1.9, 2.8, 3.4, 3.4], size=8.5,
        caption_text="表 11  reg_weight 扫描（3 任务、8 epochs）")

    para(doc, "一致性项在默认配置下只占总损失的 6–13%，属于温和约束；"
              "往小调没有收益，往大调是悬崖。所以“照搬 ANN 的 0.75”不是问题，"
              "这条待办可以关闭。", indent_first=True)

    # ------------------------------------------------------------- lesson
    heading(doc, "6  一条方法论教训")
    para(doc, "把 §5.4 的观察提炼成一句可复用的结论：", indent_first=True)
    para(doc, "**在持续学习里，任务数 ≤ 2 的消融会系统性高估任何“保护旧任务”类改动的收益。**",
         size=11.5, bold=True, align="center", color=(0x1F, 0x3B, 0x63))
    para(doc, "本项目在这个陷阱上花掉了约 4 小时 GPU：1:1 回放在 2 任务上给出 +5.6 的假信号，"
              "5 任务才暴露为 −9.1。同类风险也适用于一切“加强旧任务约束”的改动"
              "（原型约束、铰链、更高的 reg_weight）——它们在我们的 2 任务快验里全部看起来"
              "“有效或无害”，而 5 任务上全部为负。", indent_first=True)

    # ------------------------------------------------------------- paper
    heading(doc, "7  与 CLS-ER 论文的对比")
    para(doc, "论文 Table 1 的 Seq-CIFAR-10 / Class-IL / buffer 500 一栏与本项目并列：",
         indent_first=True)
    table(doc, ["方法", "class-IL", "说明"], [
        ["GEM", "26.20 ± 1.26", ""],
        ["FDR", "28.71 ± 3.23", ""],
        ["iCaRL", "47.55 ± 3.95", "也用最近类均值"],
        ["GSS", "49.73 ± 4.78", ""],
        ["**ER**", "**57.74 ± 0.27**", "纯经验回放"],
        ["Mean-ER", "70.40 ± 1.21", "两记忆平均融合"],
        ["DER", "72.70 ± 1.36", ""],
        ["**CLS-ER（论文）**", "**75.22 ± 0.71**", "论文主方法"],
        ["JOINT（上界）", "92.20 ± 0.15", "10 类联合训练"],
        ["— 本项目 —", "", ""],
        ["ANN 口径", "39.50", "介于 FDR 与 iCaRL 之间"],
        ["SNN 原生", "57.51", "≈ ER（57.74）"],
        ["SNN 原生 + NCM", "59.64", "仍低于 Mean-ER 10.8 / CLS-ER 15.6"],
    ], widths=[4.2, 4.0, 8.8], size=8.5, highlight_rows=(4, 7, 11),
        caption_text="表 12  与论文 Table 1 的并列（Class-IL，buffer 500）")

    para(doc, "三处口径差异必须说明，否则这个并列会误导：", space_after=3, keep_next=True)
    bullet(doc, "**训练预算**：论文 50 epochs，本项目 25 epochs（省一半 GPU 时间）。")
    bullet(doc, "**骨干规模**：论文 ANN ResNet-18 约 11.2 M 参数；本项目脉冲 width-32 约 2.8 M（1/4）。")
    bullet(doc, "**seed**：论文为 3 seed 均值 ± 标准差，本项目为单 seed。")
    para(doc, "因此 15.6 点的差距**不能归因于脉冲架构**。要归因，缺一个同预算、同数据划分的 "
              "ANN 骨干对照——本项目没有实现它，这是最明确的一处未完成。", indent_first=True)

    # ------------------------------------------------------------- attribution
    doc.add_page_break()
    heading(doc, "8  差距归因：稠密骨干对照")
    para(doc, "这是最初缺失、也是最关键的一个对照：**同参数量、同预算、同数据划分、同一套 "
              "CLS-ER 机制**，只把基底从脉冲换成稠密卷积。为此实现了 "
              "`backbone_ann.DenseResNet18`，它与脉冲骨干满足完全相同的接口"
              "（forward / features / forward_temporal），因此模型、双记忆、一致性损失、"
              "NCM、校准器全部一行未改。", indent_first=True)

    ann5 = r("ann_ctrl")["final"]
    annh = r("ann_ctrl")["history"]
    snnh = r("snn_e25")["history"]
    table(doc, ["", "脉冲", "稠密 ANN", "差"], [
        ["class-IL", f"{snn['cil_acc']:.2f}", f"**{ann5['cil_acc']:.2f}**",
         f"{ann5['cil_acc'] - snn['cil_acc']:+.2f}"],
        ["task-IL", f"{snn['til_acc']:.2f}", f"{ann5['til_acc']:.2f}",
         f"{ann5['til_acc'] - snn['til_acc']:+.2f}"],
        ["F_cil", f"{snn['forgetting_cil']:.2f}", f"{ann5['forgetting_cil']:.2f}",
         f"{ann5['forgetting_cil'] - snn['forgetting_cil']:+.2f}"],
        ["耗时", f"{snn['elapsed_s'] / 60:.0f} min", f"{ann5['elapsed_s'] / 60:.0f} min", "—"],
        ["参数量", "2.80 M", "2.80 M（1.00×）", "—"],
    ], widths=[2.6, 4.2, 5.4, 4.4], size=9, highlight_rows=(0,),
        caption_text="表 12b  稠密对照：只换基底，其余完全相同")

    rows = [["t0", "96.1", "98.8", "+2.8"], ["t1", "79.0", "84.0", "+5.0"],
            ["t2", "64.1", "69.1", "+5.0"], ["t3", "55.2", "61.8", "+6.6"],
            ["t4", "57.5", "**65.6**", "**+8.0**"]]
    table(doc, ["检查点", "脉冲", "稠密 ANN", "差"], rows,
          widths=[2.6, 4.2, 5.4, 4.4], size=9, highlight_rows=(4,),
          caption_text="表 12c  逐检查点：差距随任务数放大")

    para(doc, "末行 per-task：ANN `[49.9, 39.6, 64.8, 77.1, 96.6]`，脉冲 "
              "`[40.2, 31.8, 51.9, 69.8, 94.0]`——**每一个任务都高 2.6–12.9 点**。",
         indent_first=True)

    para(doc, "**归因公式（数字对得上）**", space_after=3, keep_next=True)
    code_block(doc, [
        "论文 CLS-ER（ANN, 50 epochs）          75.22",
        "   |  -9.66   <- 预算与配方（我们只有 25 epochs）",
        "   v",
        "稠密 ANN（2.80 M, 25 epochs）           65.56",
        "   |  -8.05   <- 脉冲基底（T=4 直接编码 + 时间平均放电率读出）",
        "   v",
        "脉冲 SNN（其余完全相同）                57.51",
        "---------------------------------------------",
        "总差距 17.71 = 9.66（预算）+ 8.05（基底）",
    ])

    para(doc, "三条结论：", space_after=3, keep_next=True)
    bullet(doc, "**脉冲基底有真实代价，但没有想象中大**：相对差距约 12%。"
                "“脉冲网很差”的印象里，更大的一块其实来自预算不足（9.66 点）。")
    bullet(doc, "**基底差距主要落在每个任务的绝对精度上**（低 2.6–12.9 点），F_cil 只差 3.9。"
                "这解释了此前八条“减少遗忘”的路线为何全部无效——**它们治的不是主要矛盾**。")
    bullet(doc, "**我们的稠密 ANN 也只到 65.56**，离论文还差 9.66：这一半与脉冲无关，"
                "是 25 vs 50 epochs 加上简化掉的东西。差距有两块，块块都不小。")
    para(doc, "由此，优化方向应当从“减少遗忘”转向“提高单任务表征质量”：首推 `--time-steps 8`"
              "（SD²-SNN 复现里单任务上限 80.5 → 84.1），其次 `--width 64`"
              "（把参数提到与论文同规模的 11.2 M）。", indent_first=True)

    # ------------------------------------------------------- substrate levers
    heading(doc, "8b  两个基底改进：T=8 与膜电位读出", level=2)
    t8 = r("snn_t8")["final"]
    mem = r("snn_mem2")["final"]
    table(doc, ["配置", "class-IL", "task-IL", "F_cil", "算力"], [
        ["放电率 T=4（基线）", f"{snn['cil_acc']:.2f}", f"{snn['til_acc']:.2f}",
         f"{snn['forgetting_cil']:.2f}", "1.00×"],
        ["**放电率 T=8**", f"**{t8['cil_acc']:.2f}**", f"**{t8['til_acc']:.2f}**",
         f"{t8['forgetting_cil']:.2f}", "2.08×"],
        ["**膜电位 T=4**", f"**{mem['cil_acc']:.2f}**", f"{mem['til_acc']:.2f}",
         f"{mem['forgetting_cil']:.2f}", "**1.23×**"],
        ["（参考）稠密 ANN", f"{ann5['cil_acc']:.2f}", f"{ann5['til_acc']:.2f}",
         f"{ann5['forgetting_cil']:.2f}", "0.66×"],
    ], widths=[4.6, 2.6, 2.6, 2.4, 2.4], size=9, highlight_rows=(1, 2),
        caption_text="表 12d  两个单变量基底改进（同 seed、同其余配置）")

    bullet(doc, "**T=8 是本项目最好的 SNN 结果（+1.29），但要 2 倍算力**；"
                "膜电位用 1.23× 算力换 +0.45——每点收益的成本更低（0.51× vs 0.83× 算力/点），"
                "但绝对量太小。")
    bullet(doc, "**一个假设被直接证伪**：曾推测“时间平均放电率的量化（T=4 时每维只有 9 个离散"
                "取值）是基底差距的主因”。实测完全连续的膜电位读出（96.3 / 57.96）与 9 档量化的"
                "放电率（96.1 / 57.51）几乎等价。**T=8 的收益来自“更多时间步 = 更多实际计算”，"
                "不是“更细的量化”。**")
    bullet(doc, "**剩下约 6.3 点没有被任何“改模块”的手段触动**。它更可能来自训练动力学本身"
                "（替代梯度的偏差、固定 T 的计算深度上限），要动它属于换训练范式。")

    # --------------------------------------------------------------- PLIF
    heading(doc, "8c  PLIF：把膜时间常数变成可学习（本项目最好的 SNN 结果）", level=2)
    plif = r("snn_plif")["final"]
    para(doc, "固定 `tau=2` 意味着 **17 个 LIF 层用同一个时间积分尺度**——第一层（原始像素）"
              "和最后一层（抽象特征）的泄漏速度相同，这显然不合理。PLIF（Fang et al., ICCV'21）"
              "把它改成每层一个可学习标量：", indent_first=True)
    code_block(doc, [
        "1/tau = sigmoid(a)            # a 可学习，每层一个",
        "v_t   = (1 - 1/tau) * v_{t-1} + x_t",
        "",
        "# 初始化 a 使 1/tau = 0.5（等价于原来的 tau=2）",
        "# 所以这是纯可学习化，不是换网络：只多 17 个参数",
    ])
    table(doc, ["配置", "class-IL", "task-IL", "F_cil", "算力", "相对基线"], [
        ["放电率 T=4（基线）", f"{snn['cil_acc']:.2f}", f"{snn['til_acc']:.2f}",
         f"{snn['forgetting_cil']:.2f}", "1.00×", "—"],
        ["**PLIF（可学习 tau）**", f"**{plif['cil_acc']:.2f}**", f"**{plif['til_acc']:.2f}**",
         f"{plif['forgetting_cil']:.2f}", "1.51×", f"**+{plif['cil_acc'] - snn['cil_acc']:.2f}**"],
        ["（参考）稠密 ANN", f"{ann5['cil_acc']:.2f}", f"{ann5['til_acc']:.2f}",
         f"{ann5['forgetting_cil']:.2f}", "0.66×", f"+{ann5['cil_acc'] - snn['cil_acc']:.2f}"],
    ], widths=[4.2, 2.4, 2.4, 2.2, 1.8, 2.2], size=9, highlight_rows=(1,),
        caption_text="表 12e  PLIF 的效果（同 seed、同其余配置）")

    bullet(doc, "逐检查点：**+1.7 / +3.1 / −0.1 / +2.4 / +2.2**（五个里四个为正）。")
    bullet(doc, "**task-IL 也从 88.98 涨到 91.55（+2.57）**——这一点很关键：它说明改进不是"
                "“重排 logits”，而是表征质量本身变好了。")
    bullet(doc, "补回基底差距的 **27%**（2.15 / 8.05），是所有手段里收益最高的；"
                "按“每点收益的算力成本”算也优于 T=8（0.70× vs 0.83× 算力/点）。")

    # --------------------------------------------------------------- DG
    heading(doc, "8d  齿状回模式分离（k-WTA）：假设被证伪", level=2)
    para(doc, "诊断线索是 **task-IL 89 / class-IL 57.5**——每个任务内分得开、跨任务分不开。"
              "在数学上这就是“表征不够正交”。生物靠齿状回解决这个问题：颗粒细胞比皮层输入多"
              "一个数量级（扩张），且只有 2–5% 活跃（稀疏），于是重叠的输入得到几乎不重叠的码字。"
              "而我们的实测放电率是 **5–40%**，比生物密一个数量级。", indent_first=True)

    kw = r("snn_kwta")["final"]
    dg = r("snn_dg4")["final"]
    table(doc, ["配置", "存活单元", "t0", "t1", "class-IL", "F_cil", "算力"], [
        ["无稀疏（基线）", "256（稠密）", "96.1", "79.0", f"{snn['cil_acc']:.2f}",
         f"{snn['forgetting_cil']:.2f}", "1.00×"],
        ["**k-WTA 原地稀疏（5%）**", "13", "94.2", "77.4", f"{kw['cil_acc']:.2f}",
         f"{kw['forgetting_cil']:.2f}", "1.13×"],
        ["**齿状回：4× 扩张 + 5% 稀疏**", "**51**", "**95.6**", "**78.8**",
         f"**{dg['cil_acc']:.2f}**", f"{dg['forgetting_cil']:.2f}", "1.57×"],
    ], widths=[5.4, 2.4, 1.7, 1.7, 2.2, 2.0, 1.6], size=8.5, highlight_rows=(2,),
        caption_text="表 12f  两版齿状回模型（都保持约 5% 的稀疏度）")

    bullet(doc, "**扩张修好了“信息量”这一半**：原地稀疏到 13 个单元时 t0 掉 1.9 点，"
                "扩张 4 倍后存活 51 个单元，t0 只掉 0.5 点——代价随存活数单调减小"
                "（13→56.37，51→57.41，256→57.51）。")
    bullet(doc, "**但“正交性收益”始终没有出现**：两版的 F_cil 只比基线好 0.2–0.7，在噪声内；"
                "class-IL 更是一直持平或略低。")
    bullet(doc, "**因此“表征不够正交导致跨任务混淆”这个诊断被证伪**。把码字压到 5% 稀疏"
                "（理论上两两相关性大幅下降）并没有改善 class-IL——跨任务混淆的成因不是"
                "“码字太像”，而更可能是分类器在增长的 softmax 上失效，或特征被新任务的梯度"
                "整体推走。")
    para(doc, "两个实现 bug 值得一提（都由回归测试锁住了）：读出用 `last_v.detach()` 让骨干"
              "收不到梯度（task 0 掉到 64.8）；`forward_temporal` 在扩张模式下把 256 维特征"
              "喂给了 1024 维的头（直接报形状错误）。", indent_first=True)

    # ------------------------------------------------------- combination
    doc.add_page_break()
    heading(doc, "8e  组合 PLIF + T=8：62.73，补回基底差距的 65%", level=2)
    both = r("snn_plif_t8")["final"]
    table(doc, ["配置", "class-IL", "task-IL", "F_cil", "算力", "相对基线"], [
        ["基线 T=4", f"{snn['cil_acc']:.2f}", f"{snn['til_acc']:.2f}",
         f"{snn['forgetting_cil']:.2f}", "1.00×", "—"],
        ["PLIF 单独", f"{plif['cil_acc']:.2f}", f"{plif['til_acc']:.2f}",
         f"{plif['forgetting_cil']:.2f}", "1.51×", f"+{plif['cil_acc'] - snn['cil_acc']:.2f}"],
        ["T=8 单独", f"{t8['cil_acc']:.2f}", f"{t8['til_acc']:.2f}",
         f"{t8['forgetting_cil']:.2f}", "2.08×", f"+{t8['cil_acc'] - snn['cil_acc']:.2f}"],
        ["**PLIF + T=8**", f"**{both['cil_acc']:.2f}**", f"**{both['til_acc']:.2f}**",
         f"**{both['forgetting_cil']:.2f}**", "2.10×",
         f"**+{both['cil_acc'] - snn['cil_acc']:.2f}**"],
        ["稠密 ANN（参考）", f"{ann5['cil_acc']:.2f}", f"{ann5['til_acc']:.2f}",
         f"{ann5['forgetting_cil']:.2f}", "0.66×", f"+{ann5['cil_acc'] - snn['cil_acc']:.2f}"],
    ], widths=[4.0, 2.4, 2.4, 2.2, 1.8, 2.4], size=9, highlight_rows=(3,),
        caption_text="表 12g  两个改进及其组合")

    bullet(doc, "**超可加**：2.15 + 1.29 = 3.44，而组合拿到 **5.22**，多出 1.8 点的协同。"
                "机制上讲得通——PLIF 让每层找到合适的时间尺度，T=8 提供更多时间步；"
                "**有了合适的 τ，多出来的时间步才用得上**。")
    bullet(doc, "**PLIF 在这个组合里几乎免费**：T=8 单独 225 分钟，组合 227 分钟。"
                "17 个标量参数、2 分钟开销，换来 3.9 点。")

    rows = [["t0", "96.1", "40.2", "98.0", "42.9", "98.8"], ["t1", "79.0", "31.8", "82.9", "**42.2**", "84.0"],
            ["t2", "64.1", "51.9", "66.3", "62.9", "69.1"], ["t3", "55.2", "69.8", "60.2", "71.1", "61.8"],
            ["t4", "57.5", "94.0", "**62.7**", "**94.6**", "65.6"]]
    para(doc, "**中间任务的改善是关键**（下表左半为 class-IL，右半为末行 per-task）：",
         indent_first=True)
    table(doc, ["检查点", "基线", "基线 per-task", "PLIF+T=8", "其 per-task", "ANN"], rows,
          widths=[2.0, 2.0, 3.4, 2.8, 3.2, 2.0], size=8.5, highlight_rows=(1,),
          caption_text="表 12h  逐检查点：t1 的 42.2 已经超过稠密 ANN 的 39.6")

    bullet(doc, "t1 的 per-task **42.2 已经超过稠密 ANN 的 39.6**，t2 的 62.9 也接近 ANN 的 64.8。"
                "此前所有方法在 t1/t2 都只有 22–36 之间。")
    bullet(doc, "F_cil 从 33.49 降到 **30.48**（ANN 29.60），遗忘也基本追平。")

    para(doc, "**归因账本更新**", space_after=3, keep_next=True)
    code_block(doc, [
        "论文 CLS-ER（ANN, 50ep）        75.22",
        "  -9.66   预算/配方",
        "我们的 ANN（25ep, 2.80M）        65.56",
        "  -2.83   脉冲基底   <- 原本 8.05，补回 65%",
        "我们的 SNN（PLIF + T=8）         62.73",
    ])
    para(doc, "**“脉冲基底要付 8 点代价”这个结论被推翻了**——用对架构（可学习 τ + 足够时步），"
              "代价只有 2.83 点。这同时说明前面那些“改机制”的尝试之所以无效，"
              "并不是任务本身无解，而是**改错了地方**：瓶颈在基底的训练动力学，不在持续学习的约束。",
         indent_first=True)

    # ------------------------------------------------------------- engineering
    doc.add_page_break()
    heading(doc, "9  工程问题与修复")
    para(doc, "过程中定位到 6 个会直接污染实验结果的问题，均已修复并补了回归检查：",
         indent_first=True)
    table(doc, ["#", "问题", "表现", "修复"], [
        ["1", "cosine 调度跨任务重启时继承 ≈0 学习率",
         "task 1 的 CE 从 5.08 涨到 5.15，模型完全学不动", "每个任务开始重置各组 lr"],
        ["2", "归一化特征的 EMA 原型在旋转空间里抵消",
         "原型损失卡在 0.70 ≈ log 2，特征与自己的原型正交", "改为每 epoch 从 buffer 精确重算"],
        ["3", "`--proto-weight 0` 时原型记忆不创建",
         "“只开铰链”的配置静默失效，margin 恒为 0", "两者任一为正即创建 + 断言"],
        ["4", "BiC 拟合在 `no_grad` 里跑",
         "loss 无计算图，backward 直接报错", "拟合段显式 enable_grad"],
        ["5", "in-place 列赋值切断计算图",
         "反向后 logits 与参数脱钩", "改 torch.where 的 out-of-place 写法"],
        ["6", "参数统计把两个记忆副本也算进去",
         "显示 8.39 M（实际骨干 2.80 M）", "分开报 backbone / total"],
    ], widths=[0.7, 5.2, 5.4, 5.2], size=8,
        caption_text="表 13  工程缺陷清单。其中 #1、#2 各自单独就让一批实验作废")

    # ------------------------------------------------------------- conclusion
    heading(doc, "10  结论与后续")
    para(doc, "**已确立**", bold=True, space_after=3, keep_next=True)
    bullet(doc, "CLS-ER 的机制不能直接搬到 SNN：六处失效，改造后值 +18.0 class-IL。")
    bullet(doc, "瓶颈不在分类头（NCM +2.1）、不在目标函数（六组约束全负）、"
                "不在回放比例（1:1 为 −9.1）、不在主损失权重（0.3/0.75 打平）、"
                "也不在放电率漂移（稳态校准 5 任务上 −0.25）。")
    bullet(doc, "**差距已可拆分（§8）**：17.71 点 = 9.66（预算/配方）+ 8.05（脉冲基底）。"
                "基底差距主要体现在每个任务的绝对精度（低 2.6–12.9 点），而不是遗忘。")
    bullet(doc, "**两个基底改进都只值一两点（§8b）**：T=8 拿到 +1.29（2 倍算力），"
                "膜电位读出 +0.45（1.23 倍算力）。而“读出量化是主因”这个假设被直接证伪——"
                "完全连续与 9 档量化几乎等价，说明 T 的收益来自更多实际计算，而非更细的量化。")
    bullet(doc, "**唯一划算的结构改进是 PLIF（§8c）**：每层学一个时间常数，只多 17 个参数，"
                "拿到 +2.15 class-IL 与 +2.57 task-IL。它同时改善了精度和表征质量。")
    bullet(doc, f"**两者组合达到 {r('snn_plif_t8')['final']['cil_acc']:.2f}（§8e），且超可加**："
                "单独 2.15 + 1.29，组合 5.22。PLIF 让每层找到合适的时间尺度，T=8 提供更多时间步，"
                "**有了合适的 τ，多出来的时间步才用得上**。")
    bullet(doc, "**“表征不够正交”这个诊断被证伪（§8d）**：把码字压到 5% 稀疏（并配 4 倍扩张，"
                "存活 51 个单元）之后，class-IL 仍与基线持平。跨任务混淆的成因不是码字相关，"
                "更可能是分类器在增长的 softmax 上失效、或特征被新任务梯度整体推走。")
    bullet(doc, "**最重要的反转**：脉冲基底的代价不是 8.05 点，而是 **2.83 点**。"
                "之前那些“改机制”的尝试之所以全部无效，是因为改错了地方——"
                "瓶颈在基底的训练动力学（时间常数、时间步数），不在持续学习的约束。")

    para(doc, "**未完成 / 下一步**", bold=True, space_after=3, keep_next=True)
    bullet(doc, "**width 64**（约 7 小时，显存有风险）：把参数从 2.8 M 提到与论文同规模的 11.2 M，"
                "是最后一个便宜的容量杠杆；最好在 PLIF+T=8 的基础上试。")
    bullet(doc, "**50 epochs**（约 7 小时）：验证 9.66 点的“预算差距”里有几成只是 epochs。")
    bullet(doc, "**扫 `--stable-model-alpha`**（3 组 × 20 min 快验）：α=0.999 的等效窗口只有"
                "1000 步，而一个任务是 7800 步——记忆的“记忆长度”只有任务长度的 1/8，"
                "这是唯一没调过的核心超参。")
    bullet(doc, "**多 seed**：本项目全部是单 seed，2 点以内的差异不应解读；"
                "PLIF+T=8 的 5.22 点需要 3 seed 确认。")
    bullet(doc, "**T=16**（约 7 小时）：按 T=4→8 的斜率预计再 +1 左右，收益递减，不推荐。")
    bullet(doc, "**换训练范式**（ANN-SNN 转换、长 T 训练后蒸馏到 T=4）：这 6.3 点最可能的来源是"
                "替代梯度的偏差与固定 T 的计算深度上限，动它属于新项目量级。")

    # ------------------------------------------------------------- appendix
    doc.add_page_break()
    heading(doc, "附录 A  全部运行清单")
    rows = []
    for tag, data in sorted(runs.items(), key=lambda kv: kv[0]):
        final = data["final"]
        last = data["history"][-1]["eval"]
        cil = final["cil_acc"] if final else last["cil_acc"]
        til = final["til_acc"] if final else last["til_acc"]
        fcil = final["forgetting_cil"] if final else last["forgetting_cil"]
        mins = (final["elapsed_s"] / 60) if final else None
        rows.append([tag, data["tasks_done"], fmt(cil), fmt(til), fmt(fcil),
                     f"{mins:.0f}" if mins else "—"])
    table(doc, ["run", "任务数", "class-IL", "task-IL", "F_cil", "分钟"],
          rows, widths=[7.4, 2.2, 2.6, 2.6, 2.4, 2.0], size=8,
          caption_text="表 A1  21 次训练运行（部分为快验，任务数 < 5）")

    heading(doc, "附录 B  复现命令", level=2)
    code_block(doc, [
        "# 主实验：SNN 原生（默认配置）",
        "python scripts/train.py --dataset seq_cifar10 --epochs 25 \\",
        "    --buffer-size 500 --batch-size 32 --width 32 --time-steps 4 --tag snn_e25",
        "",
        "# ANN 口径（硬选择 + logits MSE + 只 EMA 参数）",
        "python scripts/train.py ... --selector-mode hard --consistency-loss mse \\",
        "    --no-ema-buffers --tag ann_e25",
        "",
        "# 免训练的换头对比（同权重）",
        "python tools/eval_head_compare.py runs/seq_cifar10_T4_w32_buf500_snn_e25",
        "",
        "# 全部结果一览",
        "python tools/summary_table.py",
        "python tests/smoke_test.py        # 30 项自检",
    ])

    heading(doc, "附录 C  自检覆盖（30 项）", level=2)
    table(doc, ["组", "项数", "内容"], [
        ["A  CLS-ER 机制", "8", "脉冲前向、LIF 二值性、reservoir、硬/软选择器、EMA 调度、"
                               "observe、指标、跨任务 LR 复位"],
        ["B  SNN 适配", "6", "软门控连续性与 β→∞ 极限、整状态 EMA、距离函数、编码、"
                            "时序蒸馏 + 近端项、稳态阈值校准"],
        ["C  SD²-SNN 正则", "1", "分解层 / 双学习率 / Eq.18 / Eq.3 / Eq.13-14"],
        ["D  logit 校准", "6", "变换、WA 夹取、LA 边缘估计量、BiC、余弦头、任务边界重拟合"],
        ["E  NCM", "2", "原型构造与缺失类屏蔽、评测路径（含反例）"],
        ["F  原型约束", "4", "EMA 语义、损失可优化性、refresh 修复 EMA 抵消、训练集成"],
    ], widths=[3.6, 1.8, 11.4], size=8.5, numeric_from=99,
        caption_text="表 C1  tests/smoke_test.py 的覆盖范围")

    out = os.path.join(HERE, "CLS-ER+SD2-SNN_研究报告.docx")
    doc.save(out)
    print("saved", out)
    return out


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
