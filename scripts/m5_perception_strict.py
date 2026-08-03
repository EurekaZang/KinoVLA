#!/usr/bin/env python
"""M5 strict §7 perception gate on real RGB-D geometry (camera-free, CPU).

Runs the full spec §7 pipeline through the *real* pinhole back-projection (kino_vla/map/rgbd.py):
render an RGB-D frame of a ground scene → segment → unproject the depth channel into the
odometry frame → paint the costmap → physical overwrite → CLIP-similarity propagation. Unlike
scripts/isaac_m5_perception_check.py (which needs the RTX camera that segfaults on this box,
CLAUDE.md §6 #24/#26), this depends on no renderer: the geometry is genuine and a live
camera's (rgb, depth) would drop straight in.

Reports: back-projection round-trip accuracy, cross-view (odometry-frame) consistency, O7
depth-corruption displacement, and the "physics condemns the homogeneous sheet" claim — all
from pinhole-unprojected pixels rather than known world rectangles.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from kino_vla.map.appearance import cosine_similarity
from kino_vla.map.rgbd import RgbdSegmenter
from kino_vla.map.traversability_map import TraversabilityMap
from kino_vla.map.types import SemanticRegion
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect


def main() -> int:
    cfg = load_config("map/traversability_v0.yaml")
    seg = RgbdSegmenter(cfg.camera)
    lines: list[str] = ["# M5 strict §7 perception gate — real RGB-D back-projection\n"]
    checks: dict[str, bool] = {}

    intr = seg.intrinsics
    lines.append(
        f"- camera: {intr.width}x{intr.height}, fx={intr.fx:.1f}, "
        f"mount={cfg.camera.mount_height_m} m, pitch={cfg.camera.pitch_rad} rad\n"
    )

    # (1) Round-trip: an uncorrupted depth render recovers the true footprint.
    truth = Rect(2.5, 0.0, 0.6, 0.6)
    o = seg.segment(np.array([0.0, 0.0]), 0.0, [SemanticRegion(truth, "ice_sheet")])[0]
    err = math.hypot(o.footprint.cx - truth.cx, o.footprint.cy - truth.cy)
    checks["round-trip footprint < 0.1 m"] = err < 0.1
    lines.append(
        f"- (1) round-trip: true ({truth.cx},{truth.cy}) → back-proj "
        f"({o.footprint.cx:.3f},{o.footprint.cy:.3f}), error {err:.3f} m\n"
    )

    # (2) Cross-view consistency (odometry-frame invariance through real geometry).
    pose2 = np.array([0.5, -1.0])
    heading2 = math.atan2(0.0 - (-1.0), 2.5 - 0.5)
    o2 = seg.segment(pose2, heading2, [SemanticRegion(truth, "ice_sheet")])[0]
    cv = math.hypot(o.footprint.cx - o2.footprint.cx, o.footprint.cy - o2.footprint.cy)
    checks["cross-view footprint agree < 0.15 m"] = cv < 0.15
    lines.append(f"- (2) cross-view: two poses agree to {cv:.3f} m (no 'turn-around forget')\n")

    # (3) O7 depth corruption flows through the unprojection.
    bias = 0.6
    clean = seg.segment(np.array([0.0, 0.0]), 0.0, [SemanticRegion(truth, "solid_ground")])[0]
    corrupt = seg.segment(
        np.array([0.0, 0.0]), 0.0, [SemanticRegion(truth, "solid_ground", depth_bias_m=bias)]
    )[0]
    disp = corrupt.footprint.cx - clean.footprint.cx
    checks[f"O7 depth bias {bias} displaces ≈ along ray"] = disp > 0.3 and abs(disp - bias) < 0.2
    lines.append(f"- (3) O7: depth bias {bias} m → footprint displaced {disp:.3f} m outward\n")

    # (4) Material separation from rendered pixels.
    pose, heading = np.array([0.0, 0.0]), 0.0
    ei = seg.segment(pose, heading, [SemanticRegion(truth, "ice_sheet")])[0].embedding
    ei2 = seg.segment(pose2, heading2, [SemanticRegion(truth, "ice_sheet")])[0].embedding
    em = seg.segment(pose, heading, [SemanticRegion(truth, "brown_mud")])[0].embedding
    bar = float(cfg.propagation_sim_threshold)
    same, cross = cosine_similarity(ei, ei2), cosine_similarity(ei, em)
    checks[f"same-material cosine ≥ {bar}"] = same >= bar
    checks[f"cross-material cosine < {bar}"] = cross < bar
    lines.append(f"- (4) pixel appearance: ice↔ice {same:.3f} ≥ {bar} > ice↔mud {cross:.3f}\n")

    # (5) Full chain: physics condemns the homogeneous ice sheet, spares the mud.
    ice = SemanticRegion(Rect(2.6, -0.9, 0.7, 0.7), "ice_sheet")
    mud = SemanticRegion(Rect(2.6, 1.1, 0.5, 0.5), "brown_mud")
    nav = TraversabilityMap(cfg, scene=[ice, mud], segmenter=seg)
    for p, h in (
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 0.6]), -0.2),
        (np.array([0.2, -0.4]), 0.1),
    ):
        nav.observe(p, h)
    res = nav.mark_failure(np.array([2.9, -1.2]))
    far_ice = float(nav.costmap.cost_at(np.array([2.4, -0.7])))
    mud_cost = float(nav.costmap.cost_at(np.array([2.6, 1.1])))
    prop_cost = float(cfg.propagation_cost)
    checks["failure propagates to far ice cell"] = far_ice >= prop_cost
    checks["mud spared (dissimilar)"] = mud_cost < prop_cost
    lines.append(
        f"- (5) chain: stamped {res['stamped']}, propagated {res['propagated']}; "
        f"far-ice cost {far_ice:.2f} ≥ {prop_cost} > mud {mud_cost:.2f}\n"
    )

    out = Path("outputs/map/perception_strict.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines.append("\n## results\n")
    for name, ok in checks.items():
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}\n")
    out.write_text("".join(lines))

    print("".join(lines))
    passed = all(checks.values())
    print(
        "PASS: M5 §7 pipeline runs on real RGB-D back-projection"
        if passed
        else "FAIL: M5 strict perception gate"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
