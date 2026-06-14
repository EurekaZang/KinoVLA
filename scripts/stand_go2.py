"""M0 bring-up: Unitree Go2 stands on flat terrain in Isaac Lab (headless).

Exit criterion (CLAUDE.md M0): "Go2 stands in Isaac Lab under a default flat-terrain
config". The robot is spawned with its default joint configuration and held there by
the default PD actuators; we assert base height and tilt stay inside the pass band
from ``configs/sim/go2_flat.yaml`` and print a seeded trajectory hash so the same
command on the same machine is checkable for determinism.

Requires Isaac Sim + Isaac Lab and a GPU — on the CPU-only dev laptop this exits
early with a SKIP message. Written against the Isaac Lab 2.x API.

Usage (on the GPU machine):
    python scripts/stand_go2.py --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading


def _isaac_available() -> bool:
    return importlib.util.find_spec("isaaclab") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 flat-terrain stand bring-up")
    parser.add_argument("--config", default="sim/go2_flat.yaml", help="config path under configs/")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")

    if not _isaac_available():
        print("SKIP: isaaclab not importable on this machine (CPU-only dev tier).")
        print("Run on the GPU machine after the README 'GPU machine setup' steps.")
        return 0

    # AppLauncher must own the arg parser additions and start the app before any
    # other isaaclab import (isaac-lab-dev skill: always launch headless for tests).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from isaaclab_assets import UNITREE_GO2_CFG

    from kino_vla.utils.config import load_config
    from kino_vla.utils.seeding import seed_everything, trajectory_hash

    cfg = load_config(args.config, overrides={"seed": args.seed} if args.seed is not None else {})
    seed_everything(cfg.seed)

    defaults = load_config("default.yaml")
    sim_cfg = sim_utils.SimulationCfg(dt=defaults.sim.physics_dt, device=defaults.sim.device)
    sim = SimulationContext(sim_cfg)

    # Flat ground + light, then the Go2 at its default spawn state.
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0)
    )
    # Hold the default stance with a stiffer PD than the asset's RL DCMotor gains
    # (Kp=25, Kd=0.5): those assume an active locomotion policy and let the legs sag
    # and lean into a crouch under a pure static hold (configs/sim/go2_flat.yaml).
    go2_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Go2")
    go2_cfg.actuators["base_legs"].stiffness = cfg.stand_hold.stiffness
    go2_cfg.actuators["base_legs"].damping = cfg.stand_hold.damping
    robot = Articulation(go2_cfg)

    sim.reset()

    n_steps = int(cfg.episode.duration_s / defaults.sim.physics_dt)
    base_heights: list[float] = []
    base_quats: list[list[float]] = []

    for _step in range(n_steps):
        # Hold the default standing pose every physics step: PD targets = default
        # joint positions. The explicit DCMotor actuators only apply torque when
        # write_data_to_sim() runs, so gating this behind control_decimation left
        # stale torque between updates and let the stance sag/tilt below the pass
        # band; re-applying every 1 kHz step holds a clean stand.
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        sim.step()
        robot.update(defaults.sim.physics_dt)
        base_heights.append(float(robot.data.root_pos_w[0, 2]))
        base_quats.append(robot.data.root_quat_w[0].tolist())

    # Compute the verdict from data gathered during the hold *before* shutting the
    # app down: Isaac Sim 5.1's SimulationApp.close() can busy-spin and never return
    # on this headless Go2 / RTX 30xx setup (verified 2026-06-13), which would hang
    # the bring-up after the result is already decided.
    import numpy as np

    heights = np.asarray(base_heights)
    quats = np.asarray(base_quats)  # (w, x, y, z)
    # Settling transient excluded: judge the final two thirds of the hold.
    judged = slice(n_steps // 3, None)
    h_min, h_max = float(heights[judged].min()), float(heights[judged].max())
    # Tilt from quaternion: angle between body z and world z.
    w, x, y, z = quats[judged].T
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)
    tilt_max = float(np.arccos(np.clip(cos_tilt, -1.0, 1.0)).max())

    print(
        f"base height in judged window: [{h_min:.3f}, {h_max:.3f}] m "
        f"(mean {float(heights[judged].mean()):.3f})"
    )
    print(f"max tilt in judged window   : {tilt_max:.3f} rad")
    print(f"trajectory hash             : {trajectory_hash(heights, quats)}")

    ok = (
        cfg.stand_check.min_base_height_m <= h_min
        and h_max <= cfg.stand_check.max_base_height_m
        and tilt_max <= cfg.stand_check.max_tilt_rad
    )
    print("PASS: Go2 standing check" if ok else "FAIL: Go2 standing check")

    # Best-effort clean shutdown on a watchdog thread, then force-exit so the gate
    # always terminates with a deterministic code even if close() hangs (Isaac Sim
    # 5.1 teardown busy-spins here; the verdict above is already final).
    sys.stdout.flush()
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
