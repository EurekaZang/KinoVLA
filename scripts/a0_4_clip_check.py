#!/usr/bin/env python
"""A0.4 real-CLIP separability check on the collected corpus (real CLIP; §2 A0.4 / C1).

Validates the appearance library on the ACTUAL corpus snapshots the agents will see: (1) the
load-bearing CROSS-class pair (adhesion vs compliant_terrain) stays CLIP-separable, so the C1 vision
certificate (AUC≈1.0) is unaffected by the library; (2) WITHIN a class, the held-out appearances are
CLIP-distinguishable from the canonical one, so the appearance-held-out generalisation test (A2/A5)
is non-trivial (the colour shortcut is real and measurable). Uses the real ClipAppearanceEncoder +
the lane-grouped C2ST (E1 machinery).

Run:  KINOVLA_MODEL_ID=... python scripts/a0_4_clip_check.py --dir outputs/eval/a0/corpus
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _last_rgb(frames: dict, sid: str) -> np.ndarray | None:
    a = frames.get(f"{sid}__rgb")
    return None if a is None else np.asarray(a[-1], dtype=np.float32)  # (H, W, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.4 real-CLIP appearance separability check")
    ap.add_argument("--dir", default="outputs/eval/a0/corpus")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()
    d = Path(args.dir)

    from kino_vla.eval.c2st import c2st_vision
    from kino_vla.map.clip_appearance import ClipAppearanceEncoder

    lines = (d / "samples.jsonl").read_text().splitlines()
    records = [json.loads(ln) for ln in lines if ln.strip()]
    frames: dict[str, np.ndarray] = {}
    for npz in sorted(d.glob("frames_*.npz")):
        with np.load(npz) as z:
            for k in z.files:
                if k.endswith("__rgb"):
                    frames[k] = z[k]

    # group RGB by (semantic category, appearance id) for the matched pair
    by_cat_app: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for r in records:
        cat = r["ground_truth"]["category"]
        if cat not in ("adhesion", "compliant_terrain"):
            continue
        rgb = _last_rgb(frames, r["sample_id"])
        if rgb is not None:
            by_cat_app[(cat, r["appearance_id"])].append(rgb)

    enc = ClipAppearanceEncoder()

    def embed(imgs: list[np.ndarray]) -> np.ndarray:
        return enc.embed_batch(imgs)

    adhesion_apps = sorted({a for (c, a) in by_cat_app if c == "adhesion"})
    compliant_apps = sorted({a for (c, a) in by_cat_app if c == "compliant_terrain"})
    out: dict = {"adhesion_apps": adhesion_apps, "compliant_apps": compliant_apps}

    # (1) cross-class: ALL adhesion RGB vs ALL compliant RGB (C1 preserved iff AUC high)
    adh_all = [im for a in adhesion_apps for im in by_cat_app[("adhesion", a)]]
    cmp_all = [im for a in compliant_apps for im in by_cat_app[("compliant_terrain", a)]]
    if adh_all and cmp_all:
        rv = c2st_vision(embed(adh_all), embed(cmp_all), n_boot=args.n_boot, n_perm=args.n_perm)
        out["cross_class_adhesion_vs_compliant"] = {"auc": round(rv.auc, 3),
                                                    "ci": [round(x, 3) for x in rv.auc_ci]}

    # (2) canonical-only cross-class (the C1 certificate locus): yellow_board vs brown_mud
    have_canon = ("adhesion", "yellow_board") in by_cat_app and (
        ("compliant_terrain", "brown_mud") in by_cat_app)
    if have_canon:
        rv = c2st_vision(embed(by_cat_app[("adhesion", "yellow_board")]),
                         embed(by_cat_app[("compliant_terrain", "brown_mud")]),
                         n_boot=args.n_boot, n_perm=args.n_perm)
        out["canonical_yellow_vs_brown"] = {"auc": round(rv.auc, 3),
                                            "ci": [round(x, 3) for x in rv.auc_ci]}

    # (3) within-class: canonical adhesion vs each held-out adhesion (shortcut is real ⇒ AUC>0.5)
    within = {}
    canon = ("adhesion", "yellow_board")
    if canon in by_cat_app:
        for a in adhesion_apps:
            if a == "yellow_board":
                continue
            rv = c2st_vision(embed(by_cat_app[canon]), embed(by_cat_app[("adhesion", a)]),
                             n_boot=args.n_boot, n_perm=args.n_perm)
            within[f"yellow_board_vs_{a}"] = {"auc": round(rv.auc, 3),
                                              "ci": [round(x, 3) for x in rv.auc_ci]}
    out["within_class_adhesion"] = within

    xc = out.get("cross_class_adhesion_vs_compliant", {}).get("auc", 0.0)
    out["verdict"] = {
        "C1_cross_class_separable": bool(xc >= 0.85),
        "within_class_distinguishable": bool(
            within and min(v["auc"] for v in within.values()) > 0.6
        ),
    }
    (d / "a0_4_clip.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {d / 'a0_4_clip.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
