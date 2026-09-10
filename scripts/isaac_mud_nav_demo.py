"""Isaac Go2 closed-loop MUD (O2) navigation — REAL camera-grounded §7 map + VLA recovery.

Drives the physically-simulated Unitree Go2 into a brown-mud compliance patch (O2,
ComplianceField: F = k_c·s + c_c·|v| on the trunk) through the full online loop and records an
annotated MP4. Everything in the perception/recovery path is REAL (the audit fix):

  • the mud is a real material texture on the terrain (the RTX camera sees mud, not bare ground);
  • the costmap is grounded from a robot-mounted RGB-D + semantic-segmentation camera —
    LiveRtxSegmenter: the semantic mask says WHICH pixels are the hazard, real CLIP on those
    pixels says WHAT it is (512-d feature + open-vocab label), and ray∩ground says WHERE it is;
  • recovery is the trained KiNO planner (--recovery vla): it attributes the cause from the
    snapshot and picks ONE §5 primitive (attribution-driven), not the cause-blind FSM stub.

The MP4 composites, every step: the RTX chase camera (the Go2 on the textured mud), the live
camera-grounded costmap (mud marked + propagated + avoid-discs + robot trail), the live
perception-camera inset (what CLIP sees), and telemetry incl. the VLA's attribution → primitive.

One operator per Isaac process (Go2 prims persist across resets). HONEST: even fully
real, O2 may not reach the goal (Gap-3 #34c — correct attribution ≠ better outcome when the wrench
traps the robot); the deliverable is that perception + recovery are real, reported as-is.

Usage:
    python scripts/isaac_mud_nav_demo.py --headless                  # real perception + VLA
    python scripts/isaac_mud_nav_demo.py --headless --recovery fsm   # real perception + FSM stub
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading

import numpy as np


def _scenarios() -> dict:
    """name → {factory: (Rect)->(operator, scene_region), operator_name, title}.

    mud (O2): a BOUNDED soft-ground drag (crossable) → the correct recovery is push-through
    (Switch_Gait/Set_Constraint). tether (O4): a YELLOW adhesive that holds (high f_break) and
    grows with penetration → the correct recovery is the COUNTERINTUITIVE back-off + route-around
    (adhesion → Backstep). The two share matched-ish appearance-free proprioception but diverge on
    appearance (brown vs yellow) and recovery — the §8.2 P4 "vision decides" pair."""
    from kino_vla.sim.operators import ComplianceField, Tether

    def mud(r):
        op = ComplianceField(r, k_c=15.0, c_c=8.0, d_sink=0.05)  # bounded drag (crossable, §6 #38)
        return op, op.scene_region()

    def tether(r):
        # Bug-1 (user physics): the adhesive resists ONLY motion DEEPER into it — the grip grows
        # with penetration until the dog can go no further (push-through STALLS), but reversing back
        # the way it came meets ZERO resistance (peel_factor=0.0) so the dog backs out free. k
        # so the forward stall is decisive; f_break huge so it never snaps (no push-through escape).
        op = Tether(r, k=80.0, d=6.0, l0=0.0, f_break=1.0e6, peel_factor=0.0)
        return op, op.scene_region()

    return {
        "mud": {"factory": mud, "operator_name": "O2_compliance", "title": "O2 mud (push-through)"},
        "tether": {
            "factory": tether,
            "operator_name": "O4_tether",
            "title": "O4 adhesive tether (back-off + route-around)",
            # 5x the adhesive AREA: half_size [1,1] (4 m^2) -> [sqrt5, sqrt5] (20 m^2), and the goal
            # moved past the bigger patch. Per-scenario overrides ONLY (the shared demo config / the
            # run_demo regression / mud / ice are untouched). Patch x in [1.26, 5.74], goal at 7.5.
            "overrides": {
                "terrain.hazard_patch.center": [3.5, 0.0],
                "terrain.hazard_patch.half_size": [2.236, 2.236],
                "goal.pos": [7.5, 0.0],
            },
        },
    }


def _analyze_trajectory(traj: list, rect, goal, goal_tol: float, goal_reached=None) -> dict:
    """The OBJECTIVE ideal-trajectory verdict (the /goal metric), from the REAL (t,x,y) trajectory:
    did the dog ENTER the adhesive rect, EXIT it completely, route around WITHOUT a 2nd entry, and
    reach the goal? n_inside_segments == 1 means exactly one clean enter→exit (no re-entry).

    ``goal_reached`` is the loop's authoritative per-step reach (the dog may cross the goal disc
    BETWEEN the strided trajectory samples — without it the verdict can miss a true reach by a few
    mm, the run-#22 artifact). When given it wins; the CLOSEST trajectory sample is the fallback."""

    def inside(x: float, y: float) -> bool:
        return abs(x - rect.cx) <= rect.hx and abs(y - rect.cy) <= rect.hy

    flags = [inside(p[1], p[2]) for p in traj]
    segs: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    final = traj[-1] if traj else [0.0, 0.0, 0.0]
    gx, gy = float(goal[0]), float(goal[1])  # python floats (goal may be a numpy array → bool_ bug)
    min_dist = min((((p[1] - gx) ** 2 + (p[2] - gy) ** 2) ** 0.5 for p in traj), default=1e9)
    traj_reached = bool(min_dist <= goal_tol)
    reached = bool(goal_reached) if goal_reached is not None else traj_reached
    entered = bool(len(segs) >= 1)
    exited = bool(entered and not flags[-1])
    ideal = bool(entered and len(segs) == 1 and exited and reached)
    return {
        "ideal": ideal,
        "entered_adhesive": entered,
        "n_inside_segments": int(len(segs)),  # 1 = clean enter+exit; >=2 = RE-ENTERED (not ideal)
        "exited_completely": exited,
        "re_entered": bool(len(segs) >= 2),
        "reached_goal": reached,
        "inside_segments_t": [[traj[a][0], traj[b][0]] for a, b in segs],
        "patch_rect_cx_cy_hx_hy": [
            round(rect.cx, 3),
            round(rect.cy, 3),
            round(rect.hx, 3),
            round(rect.hy, 3),
        ],
        "final_xy": [round(final[1], 3), round(final[2], 3)],
        "goal_xy": [gx, gy],
    }


def _jsonable(o):  # numpy-safe JSON fallback (np.bool_ / np.float64 → python scalar)
    return o.item() if hasattr(o, "item") else str(o)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isaac Go2 mud closed-loop nav + live semantic map"
    )
    parser.add_argument("--out", default="outputs/mud_nav")
    parser.add_argument("--fps", type=int, default=20, help="output video fps")
    parser.add_argument("--stride", type=int, default=3, help="render every Nth control step")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-time", type=float, default=30.0, help="cap the episode horizon [s] for a tight clip"
    )
    parser.add_argument(
        "--recovery",
        choices=["vla", "fsm"],
        default="vla",
        help="recovery brain: the trained VLA planner (attribution-driven) or the FSM stub",
    )
    parser.add_argument("--adapter", default="outputs/vla/sft_latent/adapter_best")
    parser.add_argument("--temp", type=float, default=0.0, help="VLA sampling temperature")
    parser.add_argument(
        "--scenario",
        choices=["mud", "tether"],
        default="mud",
        help="hazard: O2 mud (push-through) or O4 adhesive tether (back-off + route-around)",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True  # RTX render for the chase camera
    simulation_app = AppLauncher(args).app

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from kino_vla.loop import run_episode
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything
    from kino_vla.vla.nav_planner import obstacle_mask

    seed = args.seed if args.seed is not None else int(load_config("default.yaml").seed)
    seed_everything(seed)

    scen = _scenarios()[args.scenario]

    # REAL camera-grounded §7 map (live_perception): a textured + semantically-tagged hazard patch
    # on the terrain, a robot-mounted RGB-D + semantic-seg camera, and the costmap grounded from
    # those real pixels (LiveRtxSegmenter: semantic mask + real-CLIP label + ray∩ground footprint).
    demo_overrides = {"max_time_s": args.max_time}
    demo_overrides.update(scen.get("overrides", {}))  # per-scenario geometry (e.g. the 5x tether)
    skeleton = build_walking_skeleton(
        seed,
        "isaac",
        demo_overrides=demo_overrides,
        record_cam=True,
        operator_factory=scen["factory"],
        live_perception=True,
    )
    nav_map = skeleton.nav_map
    seg = nav_map._segmenter  # the LiveRtxSegmenter (exposes last_rgb / last_regions)
    monitor = skeleton.monitor
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    start = np.asarray(skeleton.demo_cfg.start.pos, dtype=np.float64)

    # Recovery brain: the trained VLA planner (attributes the cause from the snapshot → ONE §5
    # primitive), or the cause-blind FSM stub. The VLA is the real fix for the "blind 90° turn".
    planner_ref = None
    if args.recovery == "vla":
        import torch

        from kino_vla.data.snapshot import SnapshotRecorder
        from kino_vla.data.taxonomy import FailureTaxonomy
        from kino_vla.monitor.reflex import ActiveProbe
        from kino_vla.shield.primitive_compiler import PrimitiveCompiler
        from kino_vla.sim.live_camera import LiveRtxCamera
        from kino_vla.vla.model import KinoVLA
        from kino_vla.vla.planner import ModelVlaPolicy, VlaPlanner

        vcfg = load_config("vla/sft.yaml")
        pcfg = load_config("data/hindsight.yaml")
        scene_region = nav_map.scene[0]
        print("[vla] loading Qwen3-VL-4B + LoRA + Kino-Projector …")
        model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
        model.eval()
        torch.set_grad_enabled(False)
        vla_policy = ModelVlaPolicy(
            model,
            pcfg,
            FailureTaxonomy(pcfg),
            route=str(vcfg.get("route", "latent")),
            n_images=int(vcfg.data.get("n_images", 1)),
            temperature=args.temp,
            proprio_detail="binned",
        )
        fsm_config = load_config("recovery/fsm_isaac.yaml")
        # The VLA's EYES: the live Isaac RTX body camera feeds every inference's RGB AND grounds its
        # waypoint picks (real intrinsics + pose) — no CPU render in the deployed path (user 06-21).
        recorder = SnapshotRecorder(
            pcfg,
            scene=[scene_region],
            operator_name=scen["operator_name"],
            appearance_class=scene_region.appearance_class,
            privileged_fn=lambda: {},
            gate_rect=None,
            live_camera=LiveRtxCamera(skeleton.backend),
        )
        planner_ref = VlaPlanner(
            cfg=fsm_config,
            policy=vla_policy,
            goal_xy=goal,
            dt=skeleton.backend.dt,
            recorder=recorder,
            compiler=PrimitiveCompiler(skeleton.shield),
            probe=ActiveProbe.from_config(fsm_config),
        )
        planner_ref.monitor = monitor  # closed-loop Backstep reads the live anomaly_score
        planner_ref.nav_map = nav_map  # Q4: Update_Topology/Backstep STAMP the costmap; Q5: read it
        policy_obj: object = planner_ref
        print("[vla] planner ready — attribution-driven recovery")
    else:
        policy_obj = skeleton.policy

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{args.scenario}_nav.mp4"
    writer = imageio.get_writer(
        str(mp4),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    # ------------------------------------------------------------------ dashboard (reused figure)
    cm = nav_map.costmap
    ox, oy = float(cm._origin[0]), float(cm._origin[1])
    res = float(cm._res)
    ny, nx = cm.shape
    extent = [ox, ox + nx * res, oy, oy + ny * res]
    # The decoupled planner's viz params — the SAME threshold/clearance A* routes around, so the
    # costmap panel draws the planner's ACTUAL obstacle set (not the retired avoid-discs, #45).
    nav_threshold = float(planner_ref.cfg.get("nav_cost_threshold", 0.5)) if planner_ref else 0.5
    nav_inflation = float(planner_ref.cfg.get("nav_inflation_m", 0.6)) if planner_ref else 0.6

    fig = plt.figure(figsize=(18, 9), dpi=100)
    gs = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.0, 1.0, 0.42],
        left=0.02,
        right=0.985,
        top=0.93,
        bottom=0.06,
        wspace=1.4,
        hspace=0.42,
    )
    ax_cam = fig.add_subplot(gs[0:2, 0:6])
    ax_map = fig.add_subplot(gs[0:2, 6:10])
    ax_clip = fig.add_subplot(gs[0, 10:12])
    ax_tel = fig.add_subplot(gs[1, 10:12])
    ax_time = fig.add_subplot(gs[2, :])  # Q6: the monitor-fire timeline (anomaly + fire markers)
    rec_txt = "VLA recovery" if args.recovery == "vla" else "FSM recovery"
    _nav_mode = getattr(planner_ref, "_nav_mode", None) if planner_ref is not None else None
    if _nav_mode == "decoupled":
        arch_txt = "DECOUPLED · VLA attributes → A* planner commits the route (#45)"
    elif _nav_mode == "vla_reactive":
        arch_txt = f"{rec_txt} · VLA per-frame nav (reactive)"
    else:
        arch_txt = rec_txt
    fig.suptitle(
        f"KiNO · Isaac Go2 · {scen['title']} · real RTX §7 map · {arch_txt}",
        fontsize=15,
        fontweight="bold",
    )
    traj: list[np.ndarray] = []
    state = {
        "n": 0,
        "last_frame": None,
        "fired": False,
        "last_code": "cruise",
        "fire_times": [],  # Q6: every monitor-fire timestamp
        "anom_hist": [],  # Q6: (t, anomaly) for the timeline
        "traj_full": [],  # GOAL: (t, x, y) every step — the objective ideal-trajectory metric
        "topo_t": None,  # #45: time of the first Update_Topology / contact mark (banner trigger)
    }

    def render(obs, event, decision, frame) -> np.ndarray:
        if frame is not None:
            state["last_frame"] = frame
        traj.append(obs.pos.copy())
        # Q6: flash the banner for ~1.2 s after EACH fire (so the strided video can't skip it)
        recent_fire = bool(state["fire_times"]) and (obs.t - state["fire_times"][-1]) < 1.2
        if decision is not None and getattr(decision, "codes", None):
            state["last_code"] = ",".join(decision.codes)

        # ---- chase camera ----
        ax_cam.cla()
        ax_cam.axis("off")
        if state["last_frame"] is not None:
            ax_cam.imshow(state["last_frame"], aspect="auto")
        else:
            ax_cam.text(0.5, 0.5, "RTX camera warming up…", ha="center", va="center")
        ax_cam.set_title(f"RTX chase camera — real Go2 · {scen['title']}", fontsize=12)
        if recent_fire:  # Q6: flash a banner for ~1.2 s after the Kino-Monitor fires
            ax_cam.text(
                0.5,
                0.95,
                f"  KINO-MONITOR FIRED  (t={state['fire_times'][-1]:.1f}s)  ",
                transform=ax_cam.transAxes,
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "round", "fc": "red", "ec": "yellow", "lw": 3, "alpha": 0.95},
            )

        # ---- live costmap (§7): the DECOUPLED planner's world (no avoid-discs, #45) ----
        ax_map.cla()
        if cm.n_physical > 0 and state["topo_t"] is None:
            state["topo_t"] = obs.t  # the first Update_Topology / contact mark — banner trigger
        recent_topo = state["topo_t"] is not None and (obs.t - state["topo_t"]) < 2.5
        # background: the §7 traversability cost map (slightly transparent so the marks read on top)
        im = ax_map.imshow(
            cm.cost_grid, origin="lower", extent=extent, cmap="RdYlGn_r",
            vmin=0.0, vmax=1.0, alpha=0.70,
        )
        # the A* OBSTACLE SET the planner ACTUALLY routes around = marked cells (cost>threshold)
        # inflated by the robot clearance — the planner's REAL input (replaces the avoid-discs).
        occ = obstacle_mask(cm.cost_grid, nav_threshold, nav_inflation, res)
        if occ.any():
            ax_map.imshow(
                np.ma.masked_where(~occ, np.ones_like(occ, dtype=float)),
                origin="lower", extent=extent, cmap=ListedColormap(["#7a00cc"]),
                vmin=0.0, vmax=1.0, alpha=0.22,
            )
        # UPDATE_TOPOLOGY: the physically CONTACT-marked cells (the VLA attributed → "physics writes
        # the map"), drawn distinct from the lower-confidence CLIP-propagated cells around them.
        phys = cm.physical_grid
        if phys.any():
            ax_map.imshow(
                np.ma.masked_where(~phys, np.ones_like(phys, dtype=float)),
                origin="lower", extent=extent, cmap=ListedColormap(["#b30000"]),
                vmin=0.0, vmax=1.0, alpha=0.55,
            )
        # ground-truth patch outline — REFERENCE ONLY: the planner NEVER sees this rect (it routes
        # off the marked costmap above). Faint dotted, so it cannot be mistaken for a planner input.
        for region in nav_map.scene:
            r = region.rect
            ax_map.add_patch(
                Rectangle((r.cx - r.hx, r.cy - r.hy), 2 * r.hx, 2 * r.hy,
                          fill=False, ec="gray", lw=1.2, ls=":")
            )
            lab = seg.last_regions[0][0] if seg.last_regions else "?"
            ax_map.text(r.cx, r.cy + r.hy + 0.15, f"CLIP: '{lab}'",
                        color="dimgray", ha="center", fontsize=8)
        # the DECOUPLED planner's live COMMITTED route around the marked costmap (#45, the money
        # shot): VLA attributes on contact → marks cells; the grid-A* planner commits this route
        # and HOLDS it (it shrinks as the dog follows it; re-planned only when new cells mark).
        if planner_ref is not None and getattr(planner_ref, "waypoints", None):
            pts = [[float(obs.pos[0]), float(obs.pos[1])]]
            pts += [[float(w[0]), float(w[1])] for w in planner_ref.waypoints]
            wp = np.array(pts)
            ax_map.plot(
                wp[:, 0],
                wp[:, 1],
                "-o",
                color="magenta",
                lw=2.4,
                ms=6,
                alpha=0.95,
                zorder=4,
            )
        # trajectory + robot + goal
        if len(traj) > 1:
            tr = np.array(traj)
            ax_map.plot(tr[:, 0], tr[:, 1], "-", color="#1f4fb0", lw=1.4, alpha=0.9)
        chx, chy = math.cos(obs.heading), math.sin(obs.heading)
        ax_map.plot(obs.pos[0], obs.pos[1], "o", color="#1f4fb0", ms=9)
        ax_map.arrow(
            obs.pos[0],
            obs.pos[1],
            0.45 * chx,
            0.45 * chy,
            head_width=0.18,
            head_length=0.16,
            fc="#1f4fb0",
            ec="#1f4fb0",
            length_includes_head=True,
        )
        ax_map.plot(goal[0], goal[1], "*", color="green", ms=16, mec="k")
        ax_map.plot(start[0], start[1], "s", color="gray", ms=7)
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        ax_map.set_aspect("equal")
        # unified legend (proxy handles — the imshow overlays don't auto-legend)
        handles = [
            Line2D([0], [0], color="magenta", marker="o", lw=2.4, label="A* committed route"),
            Patch(facecolor="#b30000", alpha=0.55, label="Update_Topology: contact-marked"),
            Patch(facecolor="#7a00cc", alpha=0.30, label="A* obstacle set (marked + clearance)"),
            Line2D([0], [0], color="gray", ls=":", label="ground-truth patch (reference)"),
        ]
        ax_map.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)
        n_phys = int(cm.n_physical)
        n_prop = max(0, int((cm.cost_grid > nav_threshold).sum()) - n_phys)
        ax_map.set_title(
            f"§7 semantic costmap · {n_phys} contact-marked + {n_prop} CLIP-propagated · "
            "A* routes around (no discs)",
            fontsize=11,
        )
        ax_map.set_xlabel("odometry x [m]")
        ax_map.set_ylabel("odometry y [m]")
        if recent_topo:  # flash the Update_Topology event (the VLA→costmap write, #45)
            ax_map.text(
                0.5, 0.965, "  UPDATE_TOPOLOGY · VLA marked the surface untraversable  ",
                transform=ax_map.transAxes, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color="white",
                bbox={"boxstyle": "round", "fc": "#7a00cc", "ec": "white", "lw": 2, "alpha": 0.95},
            )

        # ---- live perception inset: what the real RTX camera sees → CLIP ----
        ax_clip.cla()
        ax_clip.axis("off")
        if seg.last_rgb is not None:
            ax_clip.imshow(seg.last_rgb)
        reg = seg.last_regions[0] if seg.last_regions else None
        ax_clip.set_title(
            f"live RTX perception →\nCLIP '{reg[0]}' p={reg[1]:.2f}"
            if reg
            else "live RTX perception",
            fontsize=10,
        )

        # ---- telemetry ----
        ax_tel.cla()
        ax_tel.axis("off")
        spd = float(np.linalg.norm(obs.vel_body))
        cmd_spd = float(np.linalg.norm(obs.cmd_prev[:2]))
        emas = monitor.channel_emas
        if planner_ref is not None and planner_ref.decisions:
            d = planner_ref.decisions[-1]
            vla_txt = f"VLA: {d.attribution} → {d.primitive}"
        elif planner_ref is not None:
            vla_txt = "VLA: (no reflection yet)"
        else:
            vla_txt = "recovery: FSM stub (cause-blind)"
        route_txt = (
            f"planner: {len(planner_ref.waypoints)}-wp committed route [{_nav_mode}]"
            if planner_ref is not None and getattr(planner_ref, "waypoints", None)
            else "planner: (no committed route)"
        )
        lines = [
            f"t = {obs.t:5.2f} s",
            f"speed = {spd:.2f}  cmd = {cmd_spd:.2f} m/s",
            f"anomaly = {monitor.anomaly_score:.2f}",
            f"  slip {emas.get('slip_ratio', 0):.2f}  track {emas.get('tracking_error', 0):.2f}",
            f"  effort {emas.get('effort_ratio', 0):.2f}  tilt {emas.get('tilt', 0):.2f}",
            f"monitor FIRED @ t = {state['fire_times'] or '—'} s",
            vla_txt,
            route_txt,
            f"shield: {state['last_code']}",
            f"goal dist: {float(np.linalg.norm(obs.pos - goal)):.2f} m",
        ]
        ax_tel.text(
            0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace"
        )

        # ---- Q6: monitor-fire timeline (anomaly vs time, red verticals at each fire) ----
        ax_time.cla()
        hist = state["anom_hist"]
        if hist:
            ts = [h[0] for h in hist]
            an = [h[1] for h in hist]
            ax_time.plot(ts, an, "-", color="#333333", lw=1.4, label="Kino-Monitor anomaly")
            ax_time.fill_between(ts, an, color="#888", alpha=0.18)
            for ft in state["fire_times"]:
                ax_time.axvline(ft, color="red", lw=2.2, alpha=0.9)
                ax_time.text(
                    ft,
                    1.02,
                    f"FIRE t={ft:.1f}",
                    color="red",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    transform=ax_time.get_xaxis_transform(),
                )
            ax_time.axvline(obs.t, color="#1f4fb0", lw=1.6, alpha=0.7)  # the playhead (current t)
            ax_time.set_ylim(0.0, max(1.25, max(an) * 1.12))
        ax_time.set_xlim(0.0, float(args.max_time))
        ax_time.set_ylabel("anomaly", fontsize=9)
        ax_time.set_xlabel(
            "time (s)   —   RED vertical = Kino-Monitor FIRED   ·   blue = now", fontsize=10
        )
        ax_time.set_title("Monitor-fire timeline", fontsize=11, loc="left", fontweight="bold")

        if state["n"] == 0:
            fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="traversability cost [0,1]")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    # ------------------------------------------------------------------ closed loop
    def on_step(obs, event, cmd, decision) -> None:
        # Q6 + GOAL: log the anomaly, the fire times, and the REAL (t,x,y) trajectory EVERY step
        state["anom_hist"].append((round(float(obs.t), 2), float(monitor.anomaly_score)))
        state["traj_full"].append(
            [
                round(float(obs.t), 3),
                round(float(obs.pos[0]), 4),
                round(float(obs.pos[1]), 4),
                round(float(obs.heading), 3),
            ]
        )
        if event is not None:
            state["fired"] = True
            state["fire_times"].append(round(float(obs.t), 2))
        # aim the chase camera (2.8 m behind, 1.6 m up) and capture every step so RTX stays warm
        c, s = math.cos(obs.heading), math.sin(obs.heading)
        eye = np.array([obs.pos[0] - 2.8 * c, obs.pos[1] - 2.8 * s, 1.6])
        target = np.array([obs.pos[0] + 0.5 * c, obs.pos[1] + 0.5 * s, 0.25])
        skeleton.backend.aim_record_camera(eye, target)
        frame = skeleton.backend.capture_rgb()
        if state["n"] % max(1, args.stride) == 0:
            writer.append_data(render(obs, event, decision, frame))
        else:
            if frame is not None:
                state["last_frame"] = frame
            traj.append(obs.pos.copy())
        state["n"] += 1

    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        skeleton.monitor,
        policy_obj,
        skeleton.shield,
        seed=seed,
        goal_xy=goal,
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
        on_step=on_step,
        nav_map=nav_map,
        perceive_every=5,  # ~10 Hz RTX perception (quality-neutral; obstacles come from contact)
    )
    writer.close()
    plt.close(fig)

    last = seg.last_regions[0] if seg.last_regions else None
    vla_decisions = (
        [{"attribution": d.attribution, "primitive": d.primitive} for d in planner_ref.decisions]
        if planner_ref is not None
        else []
    )
    nav_log = planner_ref.nav_log if planner_ref is not None else []
    nav_sample = nav_log[:: max(1, len(nav_log) // 6)][:6] if nav_log else []

    # GOAL: the OBJECTIVE ideal-trajectory verdict from the REAL recorded (t,x,y) trajectory.
    patch_rect = nav_map.scene[0].rect
    goal_tol = float(skeleton.demo_cfg.goal.tol_m)
    traj_verdict = _analyze_trajectory(
        state["traj_full"], patch_rect, goal, goal_tol, goal_reached=bool(result.goal_reached)
    )
    (out_dir / "trajectory.json").write_text(
        json.dumps(
            {"verdict": traj_verdict, "traj": state["traj_full"]}, indent=2, default=_jsonable
        )
    )
    # a standalone trajectory plot: the path coloured inside/outside the adhesive + the patch + goal
    figt, axt = plt.subplots(figsize=(10, 6), dpi=110)
    xs = [p[1] for p in state["traj_full"]]
    ys = [p[2] for p in state["traj_full"]]
    insidef = [
        abs(p[1] - patch_rect.cx) <= patch_rect.hx and abs(p[2] - patch_rect.cy) <= patch_rect.hy
        for p in state["traj_full"]
    ]
    axt.add_patch(
        Rectangle(
            (patch_rect.cx - patch_rect.hx, patch_rect.cy - patch_rect.hy),
            2 * patch_rect.hx,
            2 * patch_rect.hy,
            fc="gold",
            ec="orange",
            alpha=0.45,
        )
    )
    axt.scatter(xs, ys, c=["red" if f else "#1f4fb0" for f in insidef], s=7, zorder=3)
    if xs:
        axt.plot(xs[0], ys[0], "ks", ms=11, label="start")
    axt.plot(goal[0], goal[1], "g*", ms=20, label="goal")
    axt.add_patch(plt.Circle((goal[0], goal[1]), goal_tol, fill=False, ec="green", ls="--"))
    axt.set_aspect("equal")
    axt.grid(True, alpha=0.3)
    axt.legend(loc="upper left")
    axt.set_title(
        f"{args.scenario}: RED=in adhesive, BLUE=out | ideal={traj_verdict['ideal']} "
        f"segs={traj_verdict['n_inside_segments']} reached={traj_verdict['reached_goal']}"
    )
    axt.set_xlabel("x (m)")
    axt.set_ylabel("y (m)")
    figt.savefig(out_dir / "trajectory.png", bbox_inches="tight")
    plt.close(figt)

    outcome = {
        "scenario": args.scenario,
        "operator": scen["operator_name"],
        "recovery": args.recovery,
        "vla_decisions": vla_decisions,
        "nav_picks": len(nav_log),  # NOMINAL 1 Hz VLA nav-picks (Replan_Waypoint, point 2)
        "nav_sample": nav_sample,  # (t, [u,v] pixel in 0..1000, [x,y] back-projected waypoint)
        "segmenter": type(nav_map._segmenter).__name__,
        "perception": "live_rtx_camera_grounded",
        "clip_label": last[0] if last else None,
        "clip_prob": round(last[1], 3) if last else None,
        "costmap_embed_dim": nav_map.costmap.embed_dim,
        "seed": seed,
        "frames_written": state["n"] // max(1, args.stride),
        "monitor_fired": result.monitor_fired,
        "fell": result.fell,
        "goal_reached": result.goal_reached,
        "final_dist_m": round(result.final_dist_m, 3),
        "sim_time_s": round(result.sim_time_s, 2),
        "n_events": len(result.events),
        "physical_cells": nav_map.costmap.n_physical,
        "avoid_discs": len(nav_map.nav_hazards()),
        "trajectory_verdict": traj_verdict,  # GOAL: enter→exit→route-around→no-reentry→goal
        "video": str(mp4),
    }
    (out_dir / "outcome.json").write_text(json.dumps(outcome, indent=2, default=_jsonable))
    print(json.dumps(outcome, indent=2, default=_jsonable))
    print(f"PASS: recorded {args.scenario} closed-loop nav → {mp4}")

    sys.stdout.flush()
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
