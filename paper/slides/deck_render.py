#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warm-paper deck rendering engine (content-free).

A small declarative layer over python-pptx + Pillow: build a slide as a list of
rect / line / text element dicts via the R/P/T/Rt/Ln/spark helpers, then render
the SAME source to BOTH an editable .pptx and pixel-accurate PNG/PDF previews
(the previews let us QA layout + look without PowerPoint/LibreOffice).

This module holds only the theme + primitives + renderers; the actual slide
content lives in the per-deck scripts that import it.
"""
import math
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------- theme
PAGE = "F5F0E6"      # warm paper background
CARD = "FCFAF5"      # light card
BAND = "EDE3D2"      # deeper warm band
TINT = "F4E2D6"      # coral-tinted band
INK = "23201B"       # near-black warm ink
INK_SOFT = "6E655A"  # secondary text
INK_FAINT = "A99F90"  # captions / page numbers
HAIR = "E2D7C5"      # hairline / divider
CORAL = "CF6F4B"     # Claude clay / terracotta
CORAL_DEEP = "A9542F"
GRAYBAR = "C2B7A6"   # muted bar (baseline comparison)
YELLOW = "E6C24E"
BROWN = "8C5A33"
SOFT1 = "EAD9C2"     # faint decorative spark
SOFT2 = "E7C9B2"

SERIF = "Noto Serif CJK SC"
SANS = "Noto Sans CJK SC"
FAMILY = {"serif": SERIF, "sans": SANS}
FONT_FILES = {
    ("serif", False): ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("serif", True): ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 0),
    ("sans", False): ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("sans", True): ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
}

PAGE_W, PAGE_H = 13.333, 7.5
ML, MR = 0.85, 0.85
CW = PAGE_W - ML - MR
DPI = 130


def hx(h):
    return RGBColor.from_string(h)


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------- data builders
def R(t, size=14.5, color=INK, font="sans", bold=False, italic=False):
    return dict(t=t, size=size, color=color, font=font, bold=bold, italic=italic)


def P(runs, align="l", sa=6.0, sb=0.0, ls=1.14):
    if isinstance(runs, dict):
        runs = [runs]
    return dict(runs=runs, align=align, sa=sa, sb=sb, ls=ls)


def T(x, y, w, h, paras, valign="t"):
    if isinstance(paras, dict):
        paras = [paras]
    return dict(type="text", x=x, y=y, w=w, h=h, paras=paras, valign=valign)


def Rt(x, y, w, h, fill=None, line=None, lw=1.0, round=0.0):
    return dict(type="rect", x=x, y=y, w=w, h=h, fill=fill, line=line, lw=lw, round=round)


def Ln(x1, y1, x2, y2, color=CORAL, w=1.5):
    return dict(type="line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, w=w)


def spark(cx, cy, r, color=CORAL, n=8, w=2.0):
    out = []
    for i in range(n):
        a = math.pi * 2 * i / n
        out.append(Ln(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a), color=color, w=w))
    return out


CIRC = ["①", "②", "③", "④"]


# ----------------------------------------------------------------------------- pptx renderer
def _noshadow(sp):
    try:
        sp.shadow.inherit = False
    except Exception:
        pass


def _setfont(run, font, _bold):
    fam = FAMILY[font]
    run.font.name = fam
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", fam)


def _add_rect(sl, e):
    shp = MSO_SHAPE.ROUNDED_RECTANGLE if e["round"] > 0 else MSO_SHAPE.RECTANGLE
    sp = sl.shapes.add_shape(shp, Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
    if e["round"] > 0:
        try:
            sp.adjustments[0] = min(0.5, e["round"] / min(e["w"], e["h"]))
        except Exception:
            pass
    if e["fill"]:
        sp.fill.solid()
        sp.fill.fore_color.rgb = hx(e["fill"])
    else:
        sp.fill.background()
    if e["line"]:
        sp.line.color.rgb = hx(e["line"])
        sp.line.width = Pt(e["lw"])
    else:
        sp.line.fill.background()
    _noshadow(sp)


def _add_line(sl, e):
    cn = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                 Inches(e["x1"]), Inches(e["y1"]), Inches(e["x2"]), Inches(e["y2"]))
    cn.line.color.rgb = hx(e["color"])
    cn.line.width = Pt(e["w"])
    _noshadow(cn)


def _add_text(sl, e):
    tb = sl.shapes.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
    tf = tb.text_frame
    tf.word_wrap = True
    try:
        tf.auto_size = MSO_AUTO_SIZE.NONE
    except Exception:
        pass
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[e["valign"]]
    for i, pa in enumerate(e["paras"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[pa["align"]]
        p.line_spacing = pa["ls"]
        p.space_after = Pt(pa["sa"])
        p.space_before = Pt(pa["sb"])
        for rn in pa["runs"]:
            r = p.add_run()
            r.text = rn["t"]
            r.font.size = Pt(rn["size"])
            r.font.bold = rn["bold"]
            r.font.italic = rn["italic"]
            r.font.color.rgb = hx(rn["color"])
            _setfont(r, rn["font"], rn["bold"])


def render_pptx(slides, path):
    prs = Presentation()
    prs.slide_width = Inches(PAGE_W)
    prs.slide_height = Inches(PAGE_H)
    blank = prs.slide_layouts[6]
    for s in slides:
        sl = prs.slides.add_slide(blank)
        bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = hx(s["bg"])
        bg.line.fill.background()
        _noshadow(bg)
        for e in s["elements"]:
            if e["type"] == "rect":
                _add_rect(sl, e)
            elif e["type"] == "line":
                _add_line(sl, e)
        for e in s["elements"]:
            if e["type"] == "text":
                _add_text(sl, e)
    prs.save(path)


# ----------------------------------------------------------------------------- pillow preview
_fcache = {}


def pil_font(font, bold, size_pt):
    key = (font, bold, round(size_pt, 1))
    if key not in _fcache:
        pth, idx = FONT_FILES[(font, bold)]
        _fcache[key] = ImageFont.truetype(pth, int(round(size_pt * DPI / 72.0)), index=idx)
    return _fcache[key]


def IN(v):
    return int(round(v * DPI))


def _p_rect(d, e):
    x0, y0 = IN(e["x"]), IN(e["y"])
    x1, y1 = IN(e["x"] + e["w"]), IN(e["y"] + e["h"])
    fill = rgb(e["fill"]) if e["fill"] else None
    line = rgb(e["line"]) if e["line"] else None
    w = max(1, int(round(e["lw"] * DPI / 72.0)))
    if e["round"] > 0:
        d.rounded_rectangle([x0, y0, x1, y1], radius=max(1, IN(e["round"])), fill=fill, outline=line, width=w)
    else:
        d.rectangle([x0, y0, x1, y1], fill=fill, outline=line, width=w)


def _p_line(d, e):
    d.line([(IN(e["x1"]), IN(e["y1"])), (IN(e["x2"]), IN(e["y2"]))],
           fill=rgb(e["color"]), width=max(1, int(round(e["w"] * DPI / 72.0))))


def _p_text(d, e):
    pad = IN(0.1)
    top_pad = IN(0.04)
    box_x = IN(e["x"]) + pad
    box_y = IN(e["y"]) + top_pad
    box_w = IN(e["w"]) - 2 * pad
    box_h = IN(e["h"])
    para_lines = []
    total_h = 0.0
    for pa in e["paras"]:
        atoms = []
        for rn in pa["runs"]:
            f = pil_font(rn["font"], rn["bold"], rn["size"])
            for ch in rn["t"]:
                atoms.append((ch, f, rn["size"], rgb(rn["color"])))
        lines = []
        cur = []
        cur_w = 0.0
        for a in atoms:
            ch, f, _sz, _c = a
            if ch == "\n":
                lines.append(cur)
                cur = []
                cur_w = 0.0
                continue
            cw = d.textlength(ch, font=f)
            if cur and cur_w + cw > box_w:
                lines.append(cur)
                cur = [a]
                cur_w = cw
            else:
                cur.append(a)
                cur_w += cw
        if cur:
            lines.append(cur)
        if not lines:
            lines = [[]]
        para_lines.append((lines, pa["ls"], pa["align"], pa["sa"]))
        for ln in lines:
            mh = max([sz for (_, _, sz, _) in ln], default=pa["runs"][0]["size"])
            total_h += mh * DPI / 72.0 * pa["ls"]
        total_h += pa["sa"] * DPI / 72.0
    if e["valign"] == "m":
        oy = box_y + max(0, (box_h - 2 * top_pad - total_h) / 2)
    elif e["valign"] == "b":
        oy = box_y + max(0, (box_h - 2 * top_pad - total_h))
    else:
        oy = box_y
    y = oy
    for (lines, ls, align, sa) in para_lines:
        for ln in lines:
            line_w = sum(d.textlength(ch, font=f) for (ch, f, _, _) in ln)
            mh = max([sz for (_, _, sz, _) in ln], default=12)
            lh = mh * DPI / 72.0 * ls
            if align == "c":
                x = box_x + max(0, (box_w - line_w) / 2)
            elif align == "r":
                x = box_x + max(0, (box_w - line_w))
            else:
                x = box_x
            for (ch, f, _sz, color) in ln:
                d.text((x, y), ch, font=f, fill=color, anchor="la")
                x += d.textlength(ch, font=f)
            y += lh
        y += sa * DPI / 72.0


def render_preview(slides, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for n, s in enumerate(slides, 1):
        img = Image.new("RGB", (IN(PAGE_W), IN(PAGE_H)), rgb(s["bg"]))
        d = ImageDraw.Draw(img)
        for e in s["elements"]:
            if e["type"] == "rect":
                _p_rect(d, e)
            elif e["type"] == "line":
                _p_line(d, e)
        for e in s["elements"]:
            if e["type"] == "text":
                _p_text(d, e)
        p = os.path.join(outdir, "slide_%d.png" % n)
        img.save(p)
        paths.append(p)
    return paths
