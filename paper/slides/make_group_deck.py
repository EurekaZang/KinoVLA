#!/usr/bin/env python
"""Build a 5-slide group-meeting deck (English, plain-language, embodied-AI audience).

Audience: embodied-AI experts, not narrowly this sub-area. So: explain every term on first use,
lead with intuition, show the strongest clean numbers, omit limitations. Embeds the real figures.

Output: paper/slides/group_meeting_5p.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path("paper/slides/group_meeting_5p.pptx")
FIG = Path("outputs/eval")
NAVY = RGBColor(0x12, 0x2A, 0x4A)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
GREY = RGBColor(0x44, 0x44, 0x44)


def _title_bar(slide, text, sub=""):
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.25), Inches(12.5), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(27)
    r.font.bold = True
    r.font.color.rgb = NAVY
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(14)
        r2.font.color.rgb = ACCENT
        r2.font.italic = True


def _bullets(slide, items, *, left=0.4, top=1.35, width=7.0, height=5.8, size=15):
    """items: list of (text, level) OR plain str (level 0). **bold** segments are emphasized."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        text = it if isinstance(it, str) else it[0]
        level = 0 if isinstance(it, str) else it[1]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        glyph = "▪  " if level == 0 else "–  "
        segs = text.split("**")
        started = False
        for i, seg in enumerate(segs):
            if not seg:
                continue
            rr = p.add_run()
            rr.text = (glyph + seg) if not started else seg
            started = True
            rr.font.size = Pt(size if level == 0 else size - 1)
            rr.font.color.rgb = NAVY if level == 0 else GREY
            rr.font.bold = (i % 2 == 1) or (level == 0 and i == 0)
        p.space_after = Pt(6)
    return box


def _pic(slide, path, *, left=7.6, top=1.5, width=5.4):
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def _footer(slide, n):
    box = slide.shapes.add_textbox(Inches(0.3), Inches(7.05), Inches(12.7), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = f"Feel It, See It, Recover  ·  Cross-Modal Failure Attribution for Safe Quadrupedal Recovery   |   {n}/5"
    r.font.size = Pt(9)
    r.font.color.rgb = GREY
    p.alignment = PP_ALIGN.RIGHT


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------- Slide 1: Title / Problem / Thesis ----------
    s = prs.slides.add_slide(blank)
    _title_bar(s, "Feel It, See It, Recover",
               "Cross-Modal Failure Attribution for Safe Quadrupedal-Navigation Recovery")
    _bullets(s, [
        "**The problem.** A legged robot walking over unknown terrain sometimes fails — it slips, "
        "gets stuck, or is overloaded. To recover *safely* it must first diagnose **why** — and the "
        "honest evidence is split across two senses that often **disagree**.",
        "**Two senses, each with blind spots.**",
        ("**Proprioception** — the robot's own body sensing (joint torques, foot slip, IMU) at "
         "**1000 Hz**: fast and always-on, but it *cannot see*.", 1),
        ("**Vision (RGB-D camera)** — reads the surface look, but is fooled by *appearance*: a patch "
         "that looks safe but is slippery ice; a glass wall that is invisible.", 1),
        ("**This work's thesis.** Resolving these cross-modal conflicts — deciding which sense to "
         "trust, *per situation* — is a distinct cognitive operation that is:", 0),
        ("(1) **necessary** — each sense has failures only the other can solve;", 1),
        ("(2) **learned** — merely plugging in a camera is not enough;", 1),
        ("(3) **consequential** — the diagnosis causally determines whether the robot falls / gets "
         "stuck / reaches the goal;", 1),
        ("(4) **generalizes** and can self-assess its own uncertainty.", 1),
        ("**Contributions (this talk):** a certified benchmark + a learned cross-modal attributor "
         "(KinoVLA) + a measured safe-recovery guarantee — all on a physically-simulated Unitree "
         "Go2 with real RTX rendering and real CLIP vision.", 0),
    ], size=15)

    # ---------- Slide 2: Necessity (C1) + taxonomy heatmap ----------
    s = prs.slides.add_slide(blank)
    _title_bar(s, "Neither Sense Suffices", "A bidirectional necessity result (Claim C1)")
    _bullets(s, [
        "We organize every failure by **which sense decides it** (the *conflict taxonomy*):",
        ("**T2 — vision-decidable:** the body-feel is *genuinely ambiguous* — two different hazards "
         "feel identical — only the camera separates them.", 1),
        ("**T3 — body-decidable:** the camera is *deceived* (slippery ice disguised as safe ground; "
         "an invisible glass wall) — only the body-feel reveals it.", 1),
        "**Constructive proof (the matched pair).** An adhesive / sticky surface (needs back-away) "
        "and soft mud (needs high-step-through) are tuned to exert the **exact same force** on the "
        "robot's legs.",
        ("⇒ the body-sensor signal is **byte-for-byte identical** (max difference = **0.0**); yet real "
         "CLIP vision separates them perfectly (AUC = **1.0**). Body-feel alone is provably blind.", 1),
        "**Mirror proof (T3).** On a patch that *looks* like safe ground but is slippery ice, vision "
        "is blind (CLIP AUC = **0.31**) while body-feel is decisive (AUC = **1.0**); on an "
        "invisible wall, body-feel detects it (**100%**), vision cannot (**0%**).",
        ("⇒ **Necessity is bidirectional**: no single modality subsumes the other. \"Feel It AND "
         "See It.\"", 0),
    ], width=7.1, size=14)
    _pic(s, FIG / "a3/heatmap.png", left=7.55, top=1.4, width=5.5)
    _footer(s, 2)

    # ---------- Slide 3: Conflict resolution is learned (C2/C5) ----------
    s = prs.slides.add_slide(blank)
    _title_bar(s, "Having Eyes Is Not Using Them", "Conflict resolution is a learned skill (Claims C2, C5)")
    _bullets(s, [
        "**Claim:** adding a camera does **not** solve conflicts — the model must be *taught* to "
        "weigh evidence against appearance.",
        "**Three-row result** (balanced accuracy — a metric immune to gaming):",
        ("**Body-only agent: 0.50** — provably capped (on a byte-identical pair it can only output "
         "one constant guess).", 1),
        ("**Four \"has-vision\" agents** (incl. a frozen CLIP+body fusion): all plateau at **0.80** "
         "— they solve the vision-decidable cases but crack nothing more.", 1),
        ("**Our full model** (latent **Kino-Tokens** + conflict **chain-of-thought**): **0.90** — "
         "the only one that breaks the ceiling. Paired McNemar test **p < 1e-5**; reproduced across "
         "**3 seeds (0.900 / 0.900 / 0.900)**.", 1),
        "**Shortcut-killer (appearance held-out):** trained on some surface looks, tested on "
        "**never-seen** looks → still **0.83** ⇒ it learned the *physics concept*, not a color cue. "
        "It even beats a frozen CLIP classifier on looks CLIP cannot separate ⇒ genuine reasoning.",
        "**Bidirectional evidence-weighing (figure):** a model trained only on \"trust-vision\" "
        "conflicts scores **0** on \"trust-body\" conflicts; trained **bidirectionally** it correctly "
        "**overrides a benign look when the body-signal is strong** (probability of \"slippery\" "
        "rises monotonically as slip grows). Genuine weighing — not \"always trust the camera.\"",
    ], width=7.1, size=13)
    _pic(s, FIG / "a3/dose_response.png", left=7.6, top=1.5, width=5.3)
    _footer(s, 3)

    # ---------- Slide 4: Attribution causes outcomes (C3) + boundary (C4) ----------
    s = prs.slides.add_slide(blank)
    _title_bar(s, "Wrong Diagnosis → Wrong Recovery",
               "Attribution causally determines safety (Claim C3) + the intervention boundary (C4)")
    _bullets(s, [
        "**The label-swap matrix M.** Hold the physical scene fixed, **force** each recovery action "
        "(a *label*), measure the real outcome on the robot — isolating the causal effect of the "
        "diagnosis. **630 real-robot episodes**, deterministic (reproducible).",
        "**The diagonal dominates:** each hazard's *correct* recovery succeeds (rate **1.0**); wrong "
        "recoveries fail (**0.0**) — statistically separated.",
        "**The headline asymmetry:**",
        ("Pushing-through (correct for **mud**) on a **sticky** surface → the robot gets **stuck** "
         "(cost **4×**);", 1),
        ("Backing-out (correct for **sticky**) on **mud** → merely **slow** (cost **1×**).", 1),
        ("⇒ mis-diagnosis cost is **asymmetric**.", 0),
        "**A safe default emerges (figure).** When unsure whether a patch is sticky or mud, "
        "**backing-out** is the expected-cost-minimizing action as soon as sticky-surface belief "
        "exceeds just **25%** — a decision-theoretic \"Safe\" rule derived from *measured "
        "consequences* (no hard-coded safety filter needed).",
        "**Intervention boundary:** the severity at which the robot's low-level adaptation stops "
        "being enough is a clean, agent-independent, reproducible number — **θ\\* = 0.275**.",
    ], width=7.2, size=13)
    _pic(s, FIG / "a4/safe_default.png", left=7.7, top=1.55, width=5.2)
    _footer(s, 4)

    # ---------- Slide 5: Generalization + abstention (C4) + system ----------
    s = prs.slides.add_slide(blank)
    _title_bar(s, "Generalizes — and Knows When It Doesn't",
               "Calibrated abstention + a strict safety advantage (Claim C4)")
    _bullets(s, [
        "**Generalization.** Attribution holds across **held-out surface appearances** "
        "(vision-dependent agents degrade most, body-only least — confirming body-feel is the "
        "appearance-invariant channel) and across **stacked operator combinations** "
        "(**100%** dominant-factor recognition).",
        "**A usable \"I'm uncertain\" signal.** A small privileged-physics predictor's "
        "prediction-error is a calibrated uncertainty score — it detects the model's own attribution "
        "errors with **AUROC 0.87**.",
        "**Abstain → safe default (figure).** When uncertain, fall back to back-away: the calibrated "
        "policy **strictly dominates** both *always-act-on-the-diagnosis* and *never-intervene*, on "
        "expected physical cost — turning the asymmetric mis-diagnosis cost into a **strict safety "
        "advantage**.",
        "**The system (real stack):** 1 kHz proprioception → **Kino-Tokens** (compressed "
        "body-signal tokens) → a **VLA recovery planner** (Qwen3-VL-4B + LoRA); evaluated on a "
        "physically-simulated Unitree Go2 + real RTX rendering + real CLIP; every number "
        "reproducible (deterministic reset).",
        "**One line:** **Feel It** (body-feel is necessary + appearance-invariant), **See It** "
        "(vision is necessary + appearance-decidable), **Recover** (the diagnosis causally "
        "determines safe recovery) — with a **calibrated escape hatch** when uncertain.",
    ], width=7.2, size=13)
    _pic(s, FIG / "a5/risk_coverage.png", left=7.7, top=1.6, width=5.2)
    _footer(s, 5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"[OK] wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
