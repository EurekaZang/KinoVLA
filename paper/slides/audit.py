#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overflow audit: for every text box, compute the wrapped height with the real
font metrics and flag boxes whose content leaves < one line of headroom
(PowerPoint clips when auto-size is off, and it wraps slightly differently from
the preview, so we want a safety margin)."""
from PIL import Image, ImageDraw
import make_deck as M

img = Image.new("RGB", (10, 10))
d = ImageDraw.Draw(img)


def required_height(e):
    pad = 0.1
    box_w_in = e["w"] - 2 * pad
    box_w = box_w_in * M.DPI
    total = 0.0
    max_line_w_ratio = 0.0
    for pa in e["paras"]:
        atoms = []
        for rn in pa["runs"]:
            f = M.pil_font(rn["font"], rn["bold"], rn["size"])
            for ch in rn["t"]:
                atoms.append((ch, f, rn["size"]))
        lines, cur, cur_w = [], [], 0.0
        for ch, f, sz in atoms:
            if ch == "\n":
                lines.append(cur); cur = []; cur_w = 0.0; continue
            cw = d.textlength(ch, font=f)
            if cur and cur_w + cw > box_w:
                lines.append(cur); cur = [(ch, f, sz)]; cur_w = cw
            else:
                cur.append((ch, f, sz)); cur_w += cw
        if cur:
            lines.append(cur)
        if not lines:
            lines = [[]]
        for ln in lines:
            mh = max([sz for (_, _, sz) in ln], default=pa["runs"][0]["size"])
            total += mh * M.DPI / 72.0 * pa["ls"]
            lw = sum(d.textlength(ch, font=f) for (ch, f, _) in ln)
            max_line_w_ratio = max(max_line_w_ratio, lw / box_w if box_w else 0)
        total += pa["sa"] * M.DPI / 72.0
    req_in = total / M.DPI + 0.10  # + top/bottom margins
    one_line_in = max(rn["size"] for pa in e["paras"] for rn in pa["runs"]) / 72.0 * 1.2
    return req_in, one_line_in, max_line_w_ratio


slides = [M.slide_cover(), M.slide2(), M.slide3(), M.slide4(), M.slide5(), M.slide6()]
flags = 0
for n, s in enumerate(slides, 1):
    for e in s["elements"]:
        if e["type"] != "text":
            continue
        req, one_line, _ = required_height(e)
        head = e["h"] - req
        if head < 0.10:  # less than ~0.1in headroom -> risk of clip in PPT
            flags += 1
            preview = e["paras"][0]["runs"][0]["t"][:22]
            print(f"slide {n}: box_h={e['h']:.2f} req={req:.2f} head={head:+.2f}  «{preview}…»")
if flags == 0:
    print("OK - every text box has >= 0.10in headroom")
else:
    print(f"\n{flags} tight box(es) above")
