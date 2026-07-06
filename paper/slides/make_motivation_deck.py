#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the KiNO *academic motivation* deck (5 slides, Chinese).

A rigorous, conference-style companion to the plain-language group-meeting deck.
Content is distilled strictly from paper/main.tex (Abstract / Introduction /
Related Work) and is meant to convey the *positioning and motivation* of the
work: the recoverability dichotomy, why it is hard, why prior systems leave the
B-class gap open, and how KiNO answers it.

Reuses the warm-paper rendering toolchain in make_deck.py (single source of
truth -> editable .pptx + PNG/PDF previews). Run with the env that has
python-pptx + Pillow (kinovla).
"""
import os

from PIL import Image

from deck_render import (
    R, P, T, Rt, Ln, spark, render_pptx, render_preview,
    PAGE, CARD, BAND, TINT, INK, INK_SOFT, INK_FAINT, HAIR,
    CORAL, CORAL_DEEP, GRAYBAR, YELLOW, BROWN, SOFT1, SOFT2,
    PAGE_W, PAGE_H, ML, MR, CW,
)

NPAGES = 5
FOOT_L = "KiNO · 面向语义失败恢复的闭环具身反思"


# --------------------------------------------------------------------------- chrome
def header(page, eyebrow, title, title_size=23):
    E = []
    E += spark(ML + 0.08, 0.67, 0.12, CORAL, 8, 1.7)
    E.append(T(ML + 0.32, 0.50, 10.5, 0.4, P(R(eyebrow, 13, CORAL, "sans", True), sa=0)))
    E.append(T(ML, 0.84, CW, 0.95, P(R(title, title_size, INK, "serif", True), ls=1.06, sa=0)))
    E.append(Ln(ML + 0.02, 1.74, ML + 1.7, 1.74, CORAL, 2.4))
    E.append(Ln(ML, 7.02, PAGE_W - MR, 7.02, HAIR, 1.0))
    E.append(T(ML, 7.08, 9.5, 0.35, P(R(FOOT_L, 10.5, INK_FAINT, "sans"), sa=0)))
    E.append(T(PAGE_W - MR - 2.0, 7.08, 2.0, 0.35,
               P(R("%d / %d" % (page, NPAGES), 10.5, INK_FAINT, "sans"), align="r", sa=0)))
    return E


def band(y, h, label, body, fill=TINT, label_color=CORAL_DEEP, size=14.0):
    """A full-width emphasis band: bold label run + body run, centered."""
    out = [Rt(ML, y, CW, h, fill=fill, round=0.06)]
    runs = [R(label, size, label_color, "sans", True)] if label else []
    runs.append(R(body, size, INK, "sans", False))
    out.append(T(ML + 0.25, y, CW - 0.5, h, P(runs, align="c", ls=1.18, sa=0), valign="m"))
    return out


# --------------------------------------------------------------------------- slide 1 cover
def slide_cover():
    E = []
    E += spark(11.55, 2.30, 1.2, SOFT1, 8, 3.0)
    E += spark(11.55, 2.30, 0.66, SOFT2, 8, 3.0)
    E += spark(1.18, 1.43, 0.13, CORAL, 8, 1.9)
    E.append(T(1.42, 1.26, 10.5, 0.4,
               P(R("机器人学 · 具身智能 · 安全控制　|　投稿目标 RSS / CoRL / ICRA / IROS", 13, CORAL, "sans", True), sa=0)))
    # hero system name
    E.append(T(1.08, 1.74, 11.4, 1.0, P(R("KiNO", 50, INK, "serif", True), ls=1.0, sa=0)))
    # full Chinese title
    E.append(T(1.10, 2.92, 11.3, 0.6,
               P(R("面向安全四足恢复的跨模态失败归因", 23, INK, "serif", True), ls=1.05, sa=0)))
    # English title
    E.append(T(1.12, 3.52, 11.2, 0.5,
               P(R("Feel It, See It, Recover: Cross-Modal Failure Attribution for Safe Quadrupedal Recovery",
                   12.5, INK_FAINT, "sans", italic=True), sa=0)))
    E.append(Ln(1.13, 4.18, 4.0, 4.18, CORAL, 2.6))
    # the central scientific question
    E.append(Rt(1.08, 4.44, PAGE_W - 2.16, 1.02, fill=TINT, round=0.06))
    E.append(T(1.34, 4.58, PAGE_W - 2.7, 0.34,
               P(R("核心科学问题", 12.5, CORAL_DEEP, "sans", True), sa=0)))
    E.append(T(1.34, 4.92, PAGE_W - 2.7, 0.48,
               P(R("哪些物理失败可由底层连续适应恢复，哪些必须依赖语义干预？", 18.5, INK, "serif", True), ls=1.05, sa=0)))
    # capability chips
    E.append(T(1.10, 5.74, 11.3, 0.5,
               P(R("宇树 Go2　·　Isaac Lab 高精度物理仿真　·　Qwen3-VL-4B + LoRA　·　CBF-QP 安全屏障　·　Kino-Fail 基准",
                   13, INK_SOFT, "sans"), sa=0)))
    E.append(Ln(1.10, 6.42, PAGE_W - 1.10, 6.42, HAIR, 1.0))
    E.append(T(1.10, 6.56, 7.5, 0.4, P(R("五层在线恢复架构 · 一快一慢双速闭环", 12.5, INK_FAINT, "sans"), sa=0)))
    E.append(T(PAGE_W - 1.10 - 6.6, 6.56, 6.6, 0.4,
               P(R("本汇报取自论文 Abstract / Introduction / Related Work", 12.5, INK_FAINT, "sans"),
                 align="r", sa=0)))
    return {"bg": PAGE, "elements": E}


# --------------------------------------------------------------------------- slide 2 recoverability dichotomy
def slide2():
    E = header(2, "研究动机 · 可恢复性二分法",
               "强化学习已让四足运动足够鲁棒——但有一类失败，连续反应无法恢复", 21)
    lx, lw = ML, 5.55

    E.append(T(lx, 2.02, lw, 0.36,
               P([R("A 类", 14.5, CORAL_DEEP, "sans", True), R("　·　连续适应可恢复", 13.5, INK, "sans", True)], sa=0)))
    E.append(T(lx, 2.42, lw, 1.3,
               P(R("均匀低摩擦、静态负载偏移、轻度黏性地面：从本体感受历史隐式推断局部动力学，在连续参数空间内调步即可。自适应控制器即可处理——我们不试图改进它。",
                   13, INK_SOFT, "sans"), ls=1.26, sa=0)))
    E.append(Ln(lx + 0.02, 3.96, lx + 1.65, 3.96, CORAL, 2.0))
    E.append(T(lx, 4.10, lw, 0.36,
               P([R("B 类", 14.5, CORAL_DEEP, "sans", True), R("　·　需离散 / 策略级决策", 13.5, INK, "sans", True)], sa=0)))
    E.append(T(lx, 4.50, lw, 1.4,
               P(R("后撤松劲、宣告整片区域不可通行、放弃力矩饱和任务并求助：正确恢复是离散、反直觉的，在连续参数空间里“加力 / 调步”只会更糟；错误归因 → 相反的恢复。",
                   13, INK_SOFT, "sans"), ls=1.26, sa=0)))

    rx = 6.95
    rw = PAGE_W - MR - rx

    def example(y, h, fill, title, tag, tag_fill, body):
        tw = 1.86
        out = [Rt(rx, y, rw, h, fill=fill, line=HAIR, lw=1.0, round=0.06)]
        out.append(Rt(rx + 0.22, y + 0.24, 0.12, 0.12, fill=CORAL))
        out.append(T(rx + 0.46, y + 0.16, rw - tw - 0.6, 0.34, P(R(title, 13.5, CORAL_DEEP, "sans", True), sa=0)))
        out.append(Rt(rx + rw - tw - 0.2, y + 0.13, tw, 0.34, fill=tag_fill, line=HAIR, lw=0.8, round=0.05))
        out.append(T(rx + rw - tw - 0.2, y + 0.13, tw, 0.34, P(R(tag, 9.5, CORAL_DEEP, "sans", True), align="c", sa=0), valign="m"))
        out.append(T(rx + 0.46, y + 0.54, rw - 0.66, h - 0.66, P(R(body, 12.0, INK, "sans"), ls=1.2, sa=0)))
        return out

    # card 1 — elastic tether: the camera cannot see it -> proprioception saves
    E += example(2.0, 1.66, CARD, "范例一 · 弹性绳缚", "视觉看不见 → 靠本体", BAND,
                 "控制器为维持指令速度而增大力矩 → 绳如被拉满的弓储能 → 最终把机器人甩失衡。安全动作是反直觉的后撤松劲，而非一味加力。")
    # card 2 — adhesive trap vs mud: feel is matched -> vision saves
    E += example(3.84, 1.82, TINT, "范例二 · 黏附陷阱 与 烂泥", "手感相同 → 靠视觉", CARD,
                 "足端突遇阻力：若为黏附陷阱 → 须后撤、标记不可通行、绕行；若为烂泥 → 须抬高步态、推进通过。两种恢复完全相反。")

    E += band(6.06, 0.82, "承上启下：",
              "正确恢复是离散、反直觉的；而判断“究竟是哪一种失败”——归因——的依据并不固定在某一种传感器上（见下页）。",
              fill=BAND, size=13.4)
    return {"bg": PAGE, "elements": E}


# --------------------------------------------------------------------------- slide 3 two-sided modality thesis
def slide3():
    E = header(3, "核心洞见",
               "没有任何单一模态能独占归因：视觉与本体，缺一不可", 22)

    gx = ML + 1.62                       # left gutter for the row labels
    gv = 0.28                            # gap between cells
    cwc = (PAGE_W - MR - gx - gv) / 2
    col_x = [gx, gx + cwc + gv]
    lbl_y = 2.18
    row_y = [2.64, 4.16]
    chh = 1.40

    E.append(T(ML, lbl_y, 1.5, 0.42, P(R("归因可分性", 11.5, INK_FAINT, "sans", True), sa=0), valign="m"))
    for cx, ch in zip(col_x, ["视觉可分 · look-different", "视觉不可分 · look-same"]):
        E.append(T(cx, lbl_y, cwc, 0.42, P(R(ch, 12.5, INK, "sans", True), align="c", sa=0), valign="m"))
    for ry, (a, b) in zip(row_y, [("本体可分", "feel-different"), ("本体不可分", "feel-same")]):
        E.append(T(ML, ry, 1.5, chh,
                   [P(R(a, 12.5, INK, "sans", True), align="c", ls=1.1, sa=3),
                    P(R(b, 10.0, INK_FAINT, "sans"), align="c", ls=1.1, sa=0)], valign="m"))

    def cell(cx, ry, hot, title, body):
        fill = TINT if hot else CARD
        line = CORAL if hot else HAIR
        out = [Rt(cx, ry, cwc, chh, fill=fill, line=line, lw=(1.5 if hot else 1.0), round=0.06)]
        out.append(T(cx + 0.2, ry + 0.15, cwc - 0.4, 0.34,
                     P(R(title, 12.8, (CORAL_DEEP if hot else INK_FAINT), "sans", True), sa=0)))
        out.append(T(cx + 0.2, ry + 0.53, cwc - 0.4, chh - 0.62,
                     P(R(body, 11.6, (INK if hot else INK_FAINT), "sans"), ls=1.2, sa=0)))
        return out

    E += cell(col_x[0], row_y[0], False, "两者皆可分",
              "简单失效，任一模态均可判别——不有趣。")
    E += cell(col_x[1], row_y[0], True, "本体是唯一依据 · 视觉被骗",
              "视觉均匀的薄冰、超载 vs 执行器衰减、相机看不见的缠绕：外观相同，只能靠手感。")
    E += cell(col_x[0], row_y[1], True, "视觉是唯一依据 · 本体被骗",
              "粘鼠板 vs 烂泥：本体签名构造为一致，只能靠看。→ 归因歧义对。")
    E += cell(col_x[1], row_y[1], False, "两者皆不可分",
              "够不到，也不声称解决。")

    E += band(5.84, 0.86, "双边论题：",
              "两个高亮格沿对角分布、永不重合；机器人无法预知失效落在哪一格 → 规划器必须同时握住视觉与本体两个通道，缺一不可。",
              fill=TINT, size=13.2)
    return {"bg": PAGE, "elements": E}


# --------------------------------------------------------------------------- slide 4 related-work gap
def slide4():
    E = header(4, "相关工作的空白",
               "现有学习系统为何留下 B 类空白", 24)
    rows = [
        ("端到端 VLA 模型", "RT-2 · OpenVLA · Octo · PaLM-E",
         "开环、低频地映射观测 → 动作，无力反馈通道；会在失败演进过程中固执执行既定动作。"),
        ("视觉闭环规划器", "CLOVER · COME-robot",
         "以相机帧率对 RGB 反应；在摩擦崩塌、执行器饱和产生显性视觉症状之前始终“盲”。"),
        ("腿式语言导航", "NaViLA · QuadrupedGPT · DoggyBot",
         "把运动当作单向黑箱技能；身体打滑或被困时，向上无任何动力学回报。"),
        ("失败推理系统", "REFLECT · AHA",
         "事后从低频视觉归因，面向固定基座机械臂，以离散文本注入结论；腿式平台缺乏维持直立的毫秒级反射。"),
    ]
    y, rh, rg = 1.98, 0.83, 0.12
    for name, refs, lim in rows:
        E.append(Rt(ML, y, CW, rh, fill=CARD, line=HAIR, lw=1.0, round=0.05))
        E.append(Rt(ML + 0.2, y + 0.18, 0.11, 0.11, fill=CORAL))
        E.append(T(ML + 0.42, y + 0.11, 3.35, 0.34, P(R(name, 13.5, INK, "sans", True), sa=0)))
        E.append(T(ML + 0.42, y + 0.45, 3.35, 0.3, P(R(refs, 10.5, INK_FAINT, "sans"), sa=0)))
        E.append(Ln(ML + 3.95, y + 0.16, ML + 3.95, y + rh - 0.16, HAIR, 1.0))
        E.append(T(ML + 4.15, y, CW - 4.35, rh, P(R(lim, 12.6, INK_SOFT, "sans"), ls=1.2, sa=0), valign="m"))
        y += rh + rg

    E += band(5.86, 0.84, "共同盲点：",
              "都缺少“高频动力学 ↔ 语义推理”的双向耦合与安全保证 — 缺(1) 从 1 kHz 动力学上行到语义规划器的通道；(2) 规划器不能破坏稳定的下行保证。",
              fill=BAND, size=13.2)
    E.append(T(ML, 6.74, CW, 0.3,
               P(R("（旁支：安全屏障文献认证的是“已知”控制器；开放词表语义地图是 视觉 → 地图，本文反向：物理 → 地图。）",
                   10.5, INK_FAINT, "sans"), sa=0)))
    return {"bg": PAGE, "elements": E}


# --------------------------------------------------------------------------- slide 5 our answer
def slide5():
    E = header(5, "我们的定位",
               "KiNO：补上缺失的上行通道与下行安全保证", 23)
    lx, lw = ML, 6.05
    E.append(T(lx, 1.98, lw, 0.34, P(R("五层在线恢复架构", 14, CORAL_DEEP, "sans", True), sa=0)))
    layers = [
        ("① 1 kHz Kino-Monitor + 反射", "计算滑移 / 跟踪误差统计，毫秒级稳住身体（刹车、增刚、降重心），绕过大模型。"),
        ("② Kino-Tokens 提取器", "特权物理蒸馏把 500 ms 本体感受窗（1D-CNN + Perceiver）压成定长 token 注入嵌入空间；附校准 OOD 分数。"),
        ("③ ~1 Hz 视觉-语言规划器（Qwen3-VL-4B+LoRA）", "融合视觉 / 文本 / Kino-Tokens 做因果归因，从扩展恢复原语库选一，维护持久语义可通行性地图。"),
        ("④ CBF 安全屏障 + 原语编译器", "执行前把每个提议原语投影到可证明安全集。"),
    ]
    yy = 2.4
    for t, b in layers:
        E.append(T(lx, yy, lw, 0.32, P(R(t, 12.8, INK, "sans", True), sa=0)))
        E.append(T(lx + 0.18, yy + 0.33, lw - 0.18, 0.7, P(R(b, 12.0, INK_SOFT, "sans"), ls=1.2, sa=0)))
        yy += 0.95

    rx = 7.35
    rw = PAGE_W - MR - rx
    E.append(T(rx, 1.98, rw, 0.34, P(R("两条设计主线", 14, CORAL_DEEP, "sans", True), sa=0)))
    E.append(Rt(rx, 2.38, rw, 1.62, fill=TINT, line=HAIR, lw=1.0, round=0.06))
    E.append(T(rx + 0.22, 2.5, rw - 0.44, 0.34, P(R("① 大模型提议，屏障函数裁决", 12.8, CORAL_DEEP, "sans", True), sa=0)))
    E.append(T(rx + 0.22, 2.86, rw - 0.44, 1.1,
               P(R("CBF-QP 最小化编辑每个原语，使捕获点不离开收缩支撑多边形 → 降阶模型下的前向不变性：无论大模型输出什么，闭环都不离开安全集。",
                   11.8, INK, "sans"), ls=1.2, sa=0)))
    E.append(Rt(rx, 4.12, rw, 1.5, fill=CARD, line=HAIR, lw=1.0, round=0.06))
    E.append(T(rx + 0.22, 4.24, rw - 0.44, 0.34, P(R("② 特权物理锚定 latent 与数据", 12.8, CORAL_DEEP, "sans", True), sa=0)))
    E.append(T(rx + 0.22, 4.60, rw - 0.44, 1.0,
               P(R("真值一致性过滤器剔除被仿真物理否定的标注；两阶段训练 —— 监督微调 + 按物理结果打分的具身偏好优化。",
                   11.8, INK, "sans"), ls=1.2, sa=0)))

    E += band(5.84, 0.86, "仿真验证（Isaac Lab · 物理仿真 Go2）：",
              "语义干预失败的归因准确率 0.97，远超调优规则状态机的 0.24；CBF-QP 安全求解 < 0.2 ms（p99）。",
              fill=BAND, size=13.2)
    return {"bg": PAGE, "elements": E}


# --------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    slides = [slide_cover(), slide2(), slide3(), slide4(), slide5()]
    pptx_path = os.path.join(here, "KiNO_动机_学术版.pptx")
    render_pptx(slides, pptx_path)
    preview_dir = os.path.join(here, "preview_motivation")
    paths = render_preview(slides, preview_dir)
    pdf_path = os.path.join(here, "KiNO_动机_学术版_preview.pdf")
    try:
        Image.init()
        imgs = [Image.open(p).convert("RGB") for p in paths]
        imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=130)
        print("PDF  :", pdf_path)
    except Exception as exc:
        print("PDF  : (skipped:", exc, ")")
    print("PPTX :", pptx_path)
    for p in paths:
        print("PNG  :", p)


if __name__ == "__main__":
    main()
