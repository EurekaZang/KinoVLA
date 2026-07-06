#!/usr/bin/env python
"""On-policy DAgger collector for nominal-nav Turn data (#43) — REAL Go2 + RTX + CLIP + VLA.

The deployed SFT VLA freezes in nominal nav (0 Turns / 94 forbidden Replan_Waypoints): the
route-around Turn data was teleport-sampled OFF-policy, so the model never saw the exact states it
actually visits (post-Backstep, the hazard filling the forward view, repeated wp_forbidden). This
script closes that distribution gap the DAgger way: it runs the CURRENT VLA planner CLOSED-LOOP on
the physical Go2 and, via the planner's read-only ``nav_trace_sink``, captures EVERY nominal nav
state the policy visits + the planner's verdict, then queries the privileged GEOMETRIC teacher
(:func:`kino_vla.vla.nav_teacher.next_nav_label`) for the correct action at that exact state — a
route-around ``Replan_Waypoint`` pixel or an in-place ``Turn``. The output is the RAW (state +
teacher action + RGB + proprio) set; scripts/build_nav_cot_dataset.py then has a real Oracle write
the reasoning CoT and filters it for geometric truth-consistency (the §10 filter, for nav).

ALL on the real stack (§0 hard rule): live Isaac RTX body camera + real CLIP map + the trained VLA.
One operator per Isaac process (Go2 prims persist across resets, #21) ⇒ run once per --scenario and
let the runs AGGREGATE into one --out (the DAgger dataset grows across scenarios + seeds + rounds).

    python scripts/collect_nav_dagger.py --headless --scenario tether --seeds 1,2,3
    python scripts/collect_nav_dagger.py --headless --scenario mud   --seeds 1,2,3   # appends
    # round R+1: point --adapter at the freshly-trained checkpoint and re-collect (on-policy again).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def _scenarios() -> dict:
    """name -> {factory:(Rect)->(op, region), operator_name, overrides, kind}.

    The full spatial anomaly-operator battery (user directive: every operator in the nav set,
    uniformly), so the VLA learns the discriminative nominal-nav behaviour per hazard TYPE:
      - TRAP operators (O4 tether, O3 collapse, O8 invisible) → after step-in + back-out the planner
        MARKS the region untraversable (map_note set) ⇒ the teacher labels a route-around Turn;
      - CROSSABLE operators (O1 ice, O2 mud) → recovered by Switch_Gait/Set_Constraint, not marked
        (map_note empty) ⇒ the teacher labels "pursue the goal / cross" (valuable NEGATIVE examples:
        do NOT route around a crossable hazard);
      - GLOBAL operators (O5 payload, O10 effort) → no spatial patch; nominal nav is goal-pursuit.
    The teacher's route-around-vs-pursue choice is driven PER-STATE by the planner's own discovery
    (the ``map_note`` signal), not by the operator label, so labels match deployment exactly."""
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.operators import (
        Collapse,
        ComplianceField,
        EffortDecay,
        InvisibleCollider,
        MuField,
        Payload,
        Tether,
    )

    def mud(r):
        op = ComplianceField(r, k_c=15.0, c_c=8.0, d_sink=0.05)
        return op, op.scene_region()

    def tether(r):
        op = Tether(r, k=80.0, d=6.0, l0=0.0, f_break=1.0e6, peel_factor=0.0)
        return op, op.scene_region()

    def ice(r):
        return MuField(region=r, mu_s=0.10, mu_d=0.08), SemanticRegion(r, "ice_sheet")

    def collapse(r):
        op = Collapse(region=r, mu_collapsed=0.09, trigger_dwell_s=0.3, mu_intact=0.8)
        return op, SemanticRegion(r, "ice_sheet")

    def invisible(r):
        return InvisibleCollider(region=r), SemanticRegion(r, "solid_ground")

    def payload(r):
        return Payload(mass_kg=16.0, com_offset_m=(0.0, 0.0)), SemanticRegion(r, "solid_ground")

    def effort(r):
        op = EffortDecay(decay_rate_per_s=0.6, floor=0.15, t_start_s=2.0)
        return op, SemanticRegion(r, "solid_ground")

    big_tether = {
        "terrain.hazard_patch.center": [3.5, 0.0],
        "terrain.hazard_patch.half_size": [2.236, 2.236],
        "goal.pos": [7.5, 0.0],
    }
    return {
        # TRAP (route-around once discovered) — the tether 5x is the #43 freeze case
        "tether": {"factory": tether, "operator_name": "O4_tether", "overrides": big_tether},
        "tether_offaxis": {
            "factory": tether,
            "operator_name": "O4_tether",
            "overrides": {**big_tether, "goal.pos": [6.5, 3.5]},  # goal off the start axis
        },
        "collapse": {"factory": collapse, "operator_name": "O3_collapse", "overrides": {}},
        "invisible": {
            "factory": invisible,
            "operator_name": "O8_invisible_collider",
            "overrides": {},
        },
        # CROSSABLE (pursue-goal / cross — NEGATIVE route-around examples)
        "mud": {"factory": mud, "operator_name": "O2_compliance", "overrides": {}},
        "ice": {"factory": ice, "operator_name": "O1_mu_field", "overrides": {}},
        # GLOBAL embodiment (no spatial patch; nominal nav = pursue goal)
        "payload": {"factory": payload, "operator_name": "O5_payload", "overrides": {}},
        "effort": {"factory": effort, "operator_name": "O10_effort_decay", "overrides": {}},
    }


def _skirt_poses(patch, goal) -> list[tuple[float, float, float]]:
    """Teleport poses that teach the POST-TURN COMMIT (#44 round-2 fix): the model spins 90° at the
    patch edge and never drives the forward skirt. These are the states it under-visits on-policy —
    (a) backout poses facing the patch (teacher ⇒ a big route-around Turn) and (b) SKIRT poses
    alongside the inflated edges facing goal-ward (teacher ⇒ a forward Replan_Waypoint = "now drive
    the skirt"). The geometric teacher labels each; (b) supplies the missing commit examples."""
    import math as _m

    gx, gy = float(goal[0]), float(goal[1])
    edge = patch.hy + 0.6  # just outside the inflated patch
    poses: list[tuple[float, float, float]] = []
    # (a) backout, facing the patch centre (forward view is the patch ⇒ teacher turns)
    for x in (0.3, 0.7, 1.1):
        for y in (-1.2, 0.0, 1.2):
            poses.append((x, y, _m.atan2(patch.cy - y, patch.cx - x)))
    # (b) SKIRT alongside the top & bottom edges, facing roughly the goal (clear ahead ⇒ waypoint)
    for x in (patch.cx - patch.hx, patch.cx, patch.cx + patch.hx):
        for sy in (+edge, -edge):
            hd = _m.atan2(gy - sy, gx - x)
            poses.append((x, sy, hd))  # facing the goal along the edge
            poses.append((x, sy, hd + _m.radians(20)))  # slight variations
            poses.append((x, sy, hd - _m.radians(20)))
    return poses


def _cruise_proprio_pool(backend, pcfg, seed: int, n: int = 8) -> list:
    """A few REAL cruise proprioception windows (clean lane) so skirt examples carry a real window
    (the nav decision is RGB-driven; proprio is the latent side channel)."""
    from kino_vla.tokens.window import RollingWindow, window_length

    snap = pcfg.snapshot
    wl = window_length(float(snap.window_ms), float(snap.control_hz))
    win = RollingWindow(wl)
    backend._start_pos = np.array([0.0, -5.5])  # clean lane clear of the patch
    backend._start_heading = 0.0
    backend.reset(seed)
    pool = []
    for k in range(120):
        obs = backend.step(np.array([0.5, 0.0, 0.0]))
        win.push(obs)
        if k >= wl and k % 8 == 0:
            pool.append(win.window().copy())
        if obs.base_height < 0.15:
            break
    return pool or [win.window().copy()]


def main() -> int:
    ap = argparse.ArgumentParser(description="On-policy DAgger nav-state collector (#43)")
    ap.add_argument("--out", default="outputs/vla/nav_dagger_raw")
    ap.add_argument(
        "--mode",
        choices=["onpolicy", "skirt"],
        default="onpolicy",
        help="onpolicy = run the VLA closed-loop (needs --adapter); skirt = teleport post-turn "
        "skirt/backout poses + teacher labels (no model, #44 post-turn-commit fix)",
    )
    ap.add_argument("--scenario", choices=list(_scenarios()), default="tether")
    ap.add_argument("--adapter", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--seeds", default="1,2,3", help="comma-separated episode seeds (on-policy DR)")
    ap.add_argument("--start-headings-deg", default="0", help="comma-separated start headings")
    ap.add_argument("--max-time", type=float, default=45.0)
    ap.add_argument("--temp", type=float, default=0.0, help="VLA sampling temperature")
    ap.add_argument("--max-per-episode", type=int, default=60, help="cap recorded states / episode")
    ap.add_argument(
        "--perceive-every",
        type=int,
        default=25,
        help="throttle the live RTX perception render+CLIP to every Nth step (~2 Hz at 25); the "
        "per-step render is the throughput bottleneck and is quality-neutral to throttle (#44)",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app  # noqa: F841 (keep the app alive for the whole run)

    import torch

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.loop import run_episode
    from kino_vla.monitor.reflex import ActiveProbe
    from kino_vla.shield.primitive_compiler import PrimitiveCompiler
    from kino_vla.sim.live_camera import LiveRtxCamera
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.nav_teacher import next_nav_label
    from kino_vla.vla.planner import ModelVlaPolicy, VlaPlanner
    from kino_vla.vla.prompt import PlannerContext, build_nav_messages

    scen = _scenarios()[args.scenario]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    headings = [math.radians(float(h)) for h in args.start_headings_deg.split(",") if h.strip()]

    vcfg = load_config("vla/sft.yaml")
    pcfg = load_config("data/hindsight.yaml")
    route = str(vcfg.get("route", "latent"))
    n_images = int(vcfg.data.get("n_images", 1))

    # One skeleton/operator per process (#21); the model loads once and is reused across episodes.
    demo_overrides = {"max_time_s": args.max_time, **scen.get("overrides", {})}
    skeleton = build_walking_skeleton(
        seeds[0],
        "isaac",
        demo_overrides=demo_overrides,
        record_cam=True,
        operator_factory=scen["factory"],
        live_perception=True,
    )
    nav_map = skeleton.nav_map
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    patches = [reg.rect for reg in nav_map.scene]
    cam = LiveRtxCamera(skeleton.backend)  # the teacher's projector (real intrinsics + pose)

    if args.mode == "skirt":
        # Teleport post-turn-commit data (no VLA needed): teacher labels at backout + SKIRT poses,
        # all DISCOVERED (map_note set ⇒ route-around). Supplies the "now drive the skirt" examples
        # the on-policy spinner under-visits (#44 round-2). Same raw schema as the on-policy path.
        from kino_vla.utils.geometry import wrap_angle
        from kino_vla.vla.prompt import format_map_note

        rect = nav_map.scene[0].rect
        poses = _skirt_poses(rect, goal)
        pool = _cruise_proprio_pool(skeleton.backend, pcfg, seeds[0])
        out = REPO_ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        meta_path, npz_path = out / "raw_meta.jsonl", out / "raw_frames.npz"
        prior_lines = meta_path.read_text().splitlines() if meta_path.exists() else []
        prior_frames = dict(np.load(npz_path)) if npz_path.exists() else {}
        records: list[dict] = []
        frames: dict[str, np.ndarray] = {}
        for seed in seeds:
            seed_everything(seed)
            for i, (x, y, hd) in enumerate(poses):
                skeleton.backend._start_pos = np.array([x, y])
                skeleton.backend._start_heading = float(hd)
                obs = skeleton.backend.reset(seed)
                if obs.base_height < 0.18 or obs.tilt > 0.6:
                    continue
                pose = obs.pos.astype(np.float64)
                heading = float(obs.heading)
                if abs(pose[0] - rect.cx) <= rect.hx and abs(pose[1] - rect.cy) <= rect.hy:
                    continue  # never label from inside the patch
                rgb = cam.snapshot_rgb()
                if rgb is None:
                    continue
                teacher = next_nav_label(
                    pose, heading, patches, goal, cam.pixel_from_world, margin=0.5
                )
                map_note = format_map_note(pose, heading, patches)  # DISCOVERED ⇒ route-around
                gb = math.degrees(
                    wrap_angle(math.atan2(goal[1] - pose[1], goal[0] - pose[0]) - heading)
                )
                ctx = PlannerContext(
                    monitor_channel="clock",
                    pose_xy=(float(pose[0]), float(pose[1])),
                    prior_outputs=[],
                    map_note=map_note,
                )
                messages = build_nav_messages(ctx, pcfg, gb, route=route, n_images=n_images)
                sid = f"skirt_{args.scenario}_{seed}_{i:03d}"
                records.append(
                    {
                        "sample_id": sid,
                        "scenario": args.scenario,
                        "operator_name": scen["operator_name"],
                        "source": "skirt",
                        "adapter": "teleport",
                        "verdict_tag": "skirt",
                        "committed": True,
                        "discovered": True,
                        "pose_xy": [round(float(pose[0]), 3), round(float(pose[1]), 3)],
                        "heading_deg": round(math.degrees(heading), 1),
                        "goal_bearing_deg": round(gb, 1),
                        "map_note": map_note,
                        "prior_outputs": [],
                        "teacher": teacher,
                        "messages": messages,
                    }
                )
                frames[f"{sid}__rgb"] = rgb[None].astype(np.float32)
                frames[f"{sid}__proprio"] = pool[i % len(pool)].astype(np.float32)
        meta_path.write_text("\n".join(prior_lines + [json.dumps(r) for r in records]))
        np.savez_compressed(npz_path, **{**prior_frames, **frames})
        n_turn = sum(r["teacher"]["kind"] == "turn" for r in records)
        (out / f"raw_card_skirt_{args.scenario}.json").write_text(
            json.dumps({"n": len(records), "turn": n_turn, "waypoint": len(records) - n_turn})
        )
        print(f"[skirt] DONE → {out}: {len(records)} states (turn={n_turn})", flush=True)
        os._exit(0)

    print("[dagger] loading Qwen3-VL-4B + LoRA + Kino-Projector …", flush=True)
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)
    tax = FailureTaxonomy(pcfg)

    records: list[dict] = []
    frames: dict[str, np.ndarray] = {}
    ep_state = {"n": 0, "last": None}

    def make_sink(seed: int, hd: float):
        ep_state["n"] = 0
        ep_state["last"] = None

        def sink(rec: dict) -> None:
            if ep_state["n"] >= args.max_per_episode:
                return
            pose = np.asarray(rec["pose_xy"], dtype=np.float64)
            heading = float(rec["heading"])
            # light on-policy dedup: skip near-identical consecutive states (same frozen pose)
            last = ep_state["last"]
            if last is not None:
                moved = float(np.linalg.norm(pose - last[0]))
                turned = abs(math.atan2(math.sin(heading - last[1]), math.cos(heading - last[1])))
                if moved < 0.1 and turned < math.radians(5) and rec["tag"] == last[2]:
                    return
            ep_state["last"] = (pose.copy(), heading, rec["tag"])
            cam.snapshot_rgb()  # refresh the cap at THIS pose so pixel_from_world is aligned
            # CONDITION the teacher on DISCOVERY (the planner sets map_note only after MARKING a
            # region untraversable — a trap escape). Pre-discovery, or a CROSSABLE hazard never
            # marked, ⇒ no patches ⇒ pursue the goal (the dog cruises in / crosses by design, #39);
            # only a DISCOVERED trap ⇒ route around (the post-backout Turn). Matches deployment.
            discovered = bool(rec["map_note"])
            patches_now = patches if discovered else []
            teacher = next_nav_label(
                pose, heading, patches_now, goal, cam.pixel_from_world, margin=0.5
            )
            rgb = np.asarray(rec["rgb"], dtype=np.float32)
            if rgb.ndim == 4:  # (n_frames, H, W, 3) ⇒ keep the most recent frame the VLA saw
                rgb = rgb[-1]
            if rgb.size == 0:
                return
            ctx = PlannerContext(
                monitor_channel="clock",
                pose_xy=(float(pose[0]), float(pose[1])),
                prior_outputs=list(rec["prior_outputs"]),
                map_note=rec["map_note"],
            )
            messages = build_nav_messages(
                ctx, pcfg, float(rec["goal_bearing_deg"]), route=route, n_images=n_images
            )
            sid = f"dagger_{args.scenario}_{seed}_{int(math.degrees(hd))}_{ep_state['n']:03d}"
            records.append(
                {
                    "sample_id": sid,
                    "scenario": args.scenario,
                    "operator_name": scen["operator_name"],
                    "source": "onpolicy",
                    "adapter": args.adapter,
                    "verdict_tag": rec["tag"],
                    "committed": bool(rec["committed"]),
                    "discovered": discovered,  # was the trap marked at this state (route-around on)
                    "pose_xy": [round(float(pose[0]), 3), round(float(pose[1]), 3)],
                    "heading_deg": round(math.degrees(heading), 1),
                    "goal_bearing_deg": round(float(rec["goal_bearing_deg"]), 1),
                    "map_note": rec["map_note"],
                    "prior_outputs": list(rec["prior_outputs"]),
                    "teacher": teacher,
                    "messages": messages,
                }
            )
            frames[f"{sid}__rgb"] = rgb[None].astype(np.float32)
            frames[f"{sid}__proprio"] = np.asarray(rec["proprio_window"], dtype=np.float32)
            ep_state["n"] += 1

        return sink

    # Incremental checkpoint: capture any prior raw set (other --scenario invocations) ONCE, then
    # rewrite prior + this run's states AFTER EVERY episode — a multi-hour run survives a kill.
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta_path, npz_path = out / "raw_meta.jsonl", out / "raw_frames.npz"
    prior_lines = meta_path.read_text().splitlines() if meta_path.exists() else []
    prior_frames = dict(np.load(npz_path)) if npz_path.exists() else {}

    def flush() -> None:
        meta_path.write_text("\n".join(prior_lines + [json.dumps(r) for r in records]))
        np.savez_compressed(npz_path, **{**prior_frames, **frames})
        card = {
            "scenario": args.scenario,
            "adapter": args.adapter,
            "seeds": seeds,
            "n_states_this_run": len(records),
            "n_turn_teacher": sum(r["teacher"]["kind"] == "turn" for r in records),
            "n_waypoint_teacher": sum(r["teacher"]["kind"] == "waypoint" for r in records),
            "n_uncommitted_freeze": sum(1 for r in records if not r["committed"]),
            "n_discovered_states": sum(1 for r in records if r["discovered"]),
            "total_states": len(prior_lines) + len(records),
        }
        (out / f"raw_card_{args.scenario}.json").write_text(json.dumps(card, indent=2))

    for seed in seeds:
        for hd in headings:
            seed_everything(seed)
            skeleton.backend._start_heading = float(hd)
            scene_region = nav_map.scene[0]
            fsm_config = load_config("recovery/fsm_isaac.yaml")
            recorder = SnapshotRecorder(
                pcfg,
                scene=[scene_region],
                operator_name=scen["operator_name"],
                appearance_class=scene_region.appearance_class,
                privileged_fn=lambda: {},
                gate_rect=None,
                live_camera=LiveRtxCamera(skeleton.backend),
            )
            planner = VlaPlanner(
                cfg=fsm_config,
                policy=ModelVlaPolicy(
                    model,
                    pcfg,
                    tax,
                    route=route,
                    n_images=n_images,
                    temperature=args.temp,
                    proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
                ),
                goal_xy=goal,
                dt=skeleton.backend.dt,
                recorder=recorder,
                compiler=PrimitiveCompiler(skeleton.shield),
                probe=ActiveProbe.from_config(fsm_config),
            )
            planner.monitor = skeleton.monitor
            planner.nav_map = nav_map
            planner.nav_trace_sink = make_sink(seed, hd)
            run_episode(
                skeleton.backend,
                skeleton.operators,
                skeleton.monitor,
                planner,
                skeleton.shield,
                seed=seed,
                goal_xy=goal,
                goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
                max_time_s=float(skeleton.demo_cfg.max_time_s),
                nav_map=nav_map,
                perceive_every=int(args.perceive_every),
            )
            flush()  # checkpoint after EVERY episode (kill-safe)
            n_disc = sum(1 for r in records if r["discovered"])
            n_turn = sum(r["teacher"]["kind"] == "turn" for r in records)
            print(
                f"[dagger] seed {seed} hd {math.degrees(hd):.0f}: {len(records)} states "
                f"(teacher_turn={n_turn} discovered={n_disc}) — checkpointed",
                flush=True,
            )

    flush()
    print(f"[dagger] DONE → {out}: {len(prior_lines) + len(records)} total states", flush=True)
    os._exit(0)  # Isaac SimulationApp.close() busy-spins (#6); data is flushed, force-exit
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
