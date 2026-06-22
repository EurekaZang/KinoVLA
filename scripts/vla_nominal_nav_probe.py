"""Offline de-risk for point-2 nominal-nav: can the recovery-tuned VLA pick a NAV waypoint?

The user's design runs the VLA at ~1 Hz: in NOMINAL mode (monitor not fired) it must output a
Replan_Waypoint (a pixel toward the goal -> back-projected to the next waypoint), and only switch to
recovery on a monitor fire. The risk: the LoRA was trained ONLY on failure snapshots (attribution +
recovery), so nominal nav-pick is zero-shot. This probe checks it CHEAPLY (one model load + a few
generations, no Isaac) through the REAL nav path (kino_vla.vla.prompt.build_nav_messages), and
reports whether the model emits Replan_Waypoint with a direction-correct pixel.

FINDING (2026-06-20): the model DOES it zero-shot — nominal -> Replan_Waypoint, direction-correct
(center for ahead, left for ahead-left, right for ahead-right). The pixel is in Qwen's 0..1000
normalised grounding space, so the planner scales by /1000 and back-projects (kino_vla.vla.planner).

    python scripts/vla_nominal_nav_probe.py
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.vla.model import KinoVLA
from kino_vla.vla.output import parse_nav_decision
from kino_vla.vla.prompt import PlannerContext, build_nav_messages


def main() -> int:
    vcfg = load_config("vla/sft.yaml")
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    print("[probe] loading Qwen3-VL-4B + LoRA + Kino-Projector ...")
    model = KinoVLA.from_pretrained(
        vcfg, device="cuda", adapter_dir="outputs/vla/sft_latent/adapter_best"
    )
    model.eval()
    torch.set_grad_enabled(False)

    rgb = np.asarray(
        Image.open(REPO_ROOT / "outputs/mud_nav/perception_probe/rgb.png").convert("RGB")
    )
    # nominal cruise window (25,11): vx,vy,yaw,cmd_vx,cmd_vy,cmd_wz,track,slip,eff,base_h,tilt
    win = np.zeros((25, 11), dtype=np.float64)
    win[:, 0], win[:, 3], win[:, 9] = 0.5, 0.6, 0.32  # cruising, nominal trunk height
    ctx = PlannerContext(monitor_channel="clock", pose_xy=(0.0, 0.0), prior_outputs=[])

    ok_nav = 0
    trials = [(0.0, 0.0), (0.7, 0.0), (0.0, 20.0), (0.0, -20.0)]  # (temperature, goal bearing deg)
    for temp, bearing in trials:
        messages = build_nav_messages(ctx, pcfg, bearing, route="latent", n_images=1)
        text = model.generate(messages, [rgb], proprio_window=win, temperature=temp)
        pd = parse_nav_decision(text, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
        px = (
            pd.annotation.primitive.params.get("point_px")
            if pd.ok and pd.annotation is not None and pd.primitive_name == "Replan_Waypoint"
            else None
        )
        # the model emits 0..1000-normalised coords; "ahead" -> u~500, "ahead-left" -> u<500, etc.
        u = float(px[0]) if px is not None and len(px) == 2 else None
        want = "center" if abs(bearing) < 8 else ("left" if bearing > 0 else "right")
        got = (
            None
            if u is None
            else ("center" if 380 < u < 620 else ("left" if u <= 380 else "right"))
        )
        ok = bool(pd.ok and pd.primitive_name == "Replan_Waypoint" and got == want)
        ok_nav += ok
        print(
            f"[temp={temp} bearing={bearing:+.0f}] parse={pd.ok} attr={pd.attribution} "
            f"prim={pd.primitive_name} px={px} want={want} got={got} -> {'NAV-OK' if ok else 'NO'}"
        )

    print(f"\nNOMINAL nav-pick (direction-correct) on {ok_nav}/{len(trials)} trials")
    print(
        "CAPABLE — the 1 Hz NOMINAL nav route works zero-shot"
        if ok_nav >= 3
        else "WEAK — nominal nav-pick may need training data (report honestly)"
    )

    # ---- HAZARD AVOIDANCE: a discovered hazard region straight ahead (the map_note the planner
    # feeds after a Backstep). The goal is AHEAD (bearing 0) but the pick MUST go to a SIDE to route
    # around the patch — this is the whole VLA-driven route-around (no geometric avoid disc).
    haz = (
        "an untraversable hazard region you got stuck in (world x [1.3,5.7], y [-2.2,2.2], bearing "
        "+0 deg) still BLOCKS the straight path to the goal. Keep going AROUND it on your current "
        "side and do NOT turn back toward the goal — never a pixel inside that x/y box — until "
        "the region is fully BEHIND you."
    )
    ctx_haz = PlannerContext(
        monitor_channel="clock", pose_xy=(0.0, 0.0), prior_outputs=[], map_note=haz
    )
    ok_avoid = 0
    avoid_trials = [0.0, 0.7, 0.7, 1.0]
    for temp in avoid_trials:
        messages = build_nav_messages(ctx_haz, pcfg, 0.0, route="latent", n_images=1)
        text = model.generate(messages, [rgb], proprio_window=win, temperature=temp)
        pd = parse_nav_decision(text, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
        px = (
            pd.annotation.primitive.params.get("point_px")
            if pd.ok and pd.annotation is not None and pd.primitive_name == "Replan_Waypoint"
            else None
        )
        u = float(px[0]) if px is not None and len(px) == 2 else None
        to_side = u is not None and (u <= 380 or u >= 620)  # NOT center ⇒ routes around the hazard
        ok_avoid += bool(to_side)
        print(
            f"[AVOID temp={temp}] parse={pd.ok} px={px} u={u} -> "
            f"{'SIDE (routes around)' if to_side else 'CENTER/none (would re-enter!)'}"
        )
    print(f"\nHAZARD-AVOIDANCE pick-to-side on {ok_avoid}/{len(avoid_trials)} trials")
    print(
        "AVOIDS — the VLA routes around a discovered hazard zero-shot"
        if ok_avoid >= 3
        else "WEAK — the VLA does NOT reliably steer around the hazard (report honestly)"
    )
    # ---- TURN PRIMITIVE: forward view is all-hazard (re-asked after a veto). The VLA must output a
    # Turn (yaw_deg) to reorient — the planner no longer turns for it (user directive: VLA owns it).
    turn_note = (
        "an untraversable hazard region (the coloured patch you got stuck in) fills the ground "
        "directly AHEAD, and the goal is straight beyond it. Any waypoint on it is REJECTED."
        " Your last pixel landed ON the forbidden hazard surface. Pick a DIFFERENT pixel on clear "
        "ground NOT on that surface, OR output a Turn (yaw_deg in [-90,90]) to rotate toward clear "
        "ground that goes around the patch."
    )
    ctx_turn = PlannerContext(
        monitor_channel="clock", pose_xy=(0.0, 0.0), prior_outputs=[], map_note=turn_note
    )
    n_turn = 0
    turn_trials = [0.0, 0.7, 0.7, 1.0]
    for temp in turn_trials:
        messages = build_nav_messages(ctx_turn, pcfg, 0.0, route="latent", n_images=1)
        text = model.generate(messages, [rgb], proprio_window=win, temperature=temp)
        pd = parse_nav_decision(text, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
        is_turn = bool(pd.ok and pd.nav_turn_deg is not None)
        n_turn += is_turn
        print(
            f"[TURN temp={temp}] parse={pd.ok} prim={pd.primitive_name} yaw={pd.nav_turn_deg} -> "
            f"{'TURN (VLA owns it)' if is_turn else 'waypoint (no turn)'}"
        )
    print(f"\nTURN-primitive used on {n_turn}/{len(turn_trials)} all-hazard-ahead trials")
    print(
        "USES TURN — the VLA emits the rotation primitive when stuck"
        if n_turn >= 1
        else "WEAK — the VLA never turned; the planner-fallback removal may strand it (be honest)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
