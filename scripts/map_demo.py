#!/usr/bin/env python
"""M5 semantic-traversability-map demo (spec §7) — the runnable M5 deliverable.

Three parts, each a spec §7 / CLAUDE.md M5 exit criterion, then a combined report:

  A. Turn-around persistence — a region marked untraversable stays marked after the
     robot rotates 360° and re-approaches (odometry-frame memory, not view-relative).
  B. Propagation — stepping through one thin-ice cell down-weights the whole visually
     homogeneous sheet (CLIP-similarity label propagation).
  C. Map-served planner — a closed-loop episode where the map observes, a physical
     failure overwrites the costmap, the mark propagates, and the resulting avoid discs
     route the recovery planner around the entire hazard region.

Writes outputs/map/map_demo.md and prints PASS/FAIL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kino_vla.loop import run_episode
from kino_vla.map import SemanticRegion, TraversabilityMap
from kino_vla.skeleton import build_walking_skeleton
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.utils.geometry import Rect

MAP_CFG = load_config("map/traversability_v0.yaml")


def part_a_persistence() -> tuple[bool, str]:
    scene = [SemanticRegion(Rect(3.0, 0.0, 1.0, 1.0), "ice_sheet")]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    tm.observe(np.array([0.0, 0.0]), 0.0)
    tm.mark_failure(np.array([3.0, 0.0]))
    cost_before = tm.costmap.cost_at(np.array([3.0, 0.0]))
    # Rotate in place through a full turn, re-observing from every heading.
    for k in range(24):
        heading = 2.0 * np.pi * k / 24.0
        tm.observe(np.array([0.0, 0.0]), heading)
    cost_after = tm.costmap.cost_at(np.array([3.0, 0.0]))
    ok = cost_before >= 0.99 and cost_after >= 0.99
    return ok, f"cost before 360° = {cost_before:.2f}, after = {cost_after:.2f} (sticky)"


def part_b_propagation() -> tuple[bool, str]:
    # A wide homogeneous ice sheet + a distinct dry-concrete patch elsewhere.
    scene = [
        SemanticRegion(Rect(3.5, 0.0, 1.5, 1.5), "ice_sheet"),
        SemanticRegion(Rect(3.5, 4.0, 1.0, 1.0), "dry_concrete"),
    ]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    # Observe broadly (look around so both regions enter the costmap).
    for heading in np.linspace(-np.pi, np.pi, 16):
        tm.observe(np.array([0.0, 0.0]), heading)
    # Step through ONE cell of the ice sheet.
    res = tm.mark_failure(np.array([2.2, 0.0]))
    far_ice = tm.costmap.cost_at(np.array([4.8, 1.2]))  # far corner of the same sheet
    concrete = tm.costmap.cost_at(np.array([3.5, 4.0]))  # the dissimilar patch
    ok = far_ice >= MAP_CFG.propagation_cost - 1e-9 and concrete < 0.1
    return ok, (
        f"propagated {res['propagated']} cells; far ice cell cost {far_ice:.2f}, "
        f"dry-concrete cost {concrete:.2f} (sheet down-weighted, concrete untouched)"
    )


def part_c_closed_loop() -> tuple[bool, str]:
    # The exact walking-skeleton scenario (O1 ice on the demo terrain) with the semantic
    # map wired in: the map observes the visible ice patch, the on-ground slip overwrites
    # the costmap, the mark propagates over the homogeneous "ice_sheet", and the avoid discs
    # feed the recovery planner — which still routes to the goal (the map augments, not
    # breaks, the M1-green detour).
    seed = 42
    sk = build_walking_skeleton(seed)
    goal = np.asarray(sk.demo_cfg.goal.pos, dtype=np.float64)
    patch = sk.terrain.hazard_patch
    scene = [SemanticRegion(Rect(patch.cx, patch.cy, patch.hx, patch.hy), "ice_sheet")]
    nav_map = TraversabilityMap(MAP_CFG, scene=scene)
    result = run_episode(
        sk.backend,
        sk.operators,
        sk.monitor,
        sk.policy,
        sk.shield,
        seed=seed,
        goal_xy=goal,
        goal_tol_m=float(sk.demo_cfg.goal.tol_m),
        max_time_s=float(sk.demo_cfg.max_time_s),
        nav_map=nav_map,
    )
    n_phys = nav_map.costmap.n_physical
    n_hazards = len(nav_map.nav_hazards())
    ok = result.goal_reached and not result.fell and result.monitor_fired and n_phys > 0
    return ok, (
        f"goal_reached={result.goal_reached} fell={result.fell} "
        f"monitor_fired={result.monitor_fired} physical_cells={n_phys} "
        f"avoid_discs={n_hazards} detours={sk.policy.backstep_count}"
    )


def main() -> int:
    parts = [
        ("A. turn-around persistence", part_a_persistence),
        ("B. homogeneous-region propagation", part_b_propagation),
        ("C. map-served closed-loop planner", part_c_closed_loop),
    ]
    lines = ["# M5 Semantic Traversability Map — Demo (spec §7)", ""]
    all_ok = True
    for name, fn in parts:
        ok, detail = fn()
        all_ok = all_ok and ok
        lines.append(f"- **{name}**: {'PASS' if ok else 'FAIL'} — {detail}")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    lines += ["", f"**{'PASS' if all_ok else 'FAIL'}: M5 map demo**", ""]
    out = Path(REPO_ROOT) / "outputs" / "map" / "map_demo.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[map_demo] wrote {out}")
    print("PASS: M5 map demo" if all_ok else "FAIL: M5 map demo")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
