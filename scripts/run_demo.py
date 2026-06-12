"""Walking-skeleton demo — the permanent M1 deliverable (CLAUDE.md §1).

Go2 walks toward the goal, crosses onto an O1 ice patch, the Kino-Monitor fires,
the scripted FSM backsteps and replans a detour, the pass-through shield forwards
every command, and the robot reaches the goal without falling.

Pinned output contract (asserted by tests/test_demo.py and CI):
    monitor fired: True ...
    fall: False
    goal reached: True ...
    PASS: walking skeleton demo

Backends:
    --backend surrogate  CPU planar model — runs anywhere, used by CI.
    --backend isaac      Isaac Lab Go2 (GPU; kinematic root drive until the M2
                         locomotion policy lands). Verified on the RTX 5090 box.
    --backend auto       isaac when isaaclab is importable, else surrogate.

Usage:
    python scripts/run_demo.py --backend surrogate
    python scripts/run_demo.py --backend isaac --headless   # GPU machine
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


def _isaac_available() -> bool:
    return importlib.util.find_spec("isaaclab") is not None


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=["auto", "surrogate", "isaac"], default="auto")
    pre_args, _ = pre.parse_known_args()
    backend = pre_args.backend
    if backend == "auto":
        backend = "isaac" if _isaac_available() else "surrogate"

    parser = argparse.ArgumentParser(description="Kino-VLA walking skeleton demo")
    parser.add_argument("--backend", choices=["auto", "surrogate", "isaac"], default="auto")
    parser.add_argument("--seed", type=int, default=None, help="override configs/default.yaml")

    simulation_app = None
    if backend == "isaac":
        if not _isaac_available():
            print("FAIL: --backend isaac requested but isaaclab is not importable.")
            return 1
        # AppLauncher owns --headless and must start before other isaaclab imports.
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        simulation_app = AppLauncher(args).app
    else:
        parser.add_argument("--headless", action="store_true", help="no-op on the surrogate")
        args = parser.parse_args()

    from kino_vla.skeleton import run_walking_skeleton
    from kino_vla.utils.config import load_config
    from kino_vla.utils.seeding import seed_everything

    seed = args.seed if args.seed is not None else int(load_config("default.yaml").seed)
    seed_everything(seed)
    result, skeleton = run_walking_skeleton(seed, backend)
    if simulation_app is not None:
        simulation_app.close()

    first = result.events[0] if result.events else None
    patch = skeleton.terrain.hazard_patch
    print(f"backend: {backend} seed={seed}")
    print(
        f"ice patch: center=({patch.cx:.2f},{patch.cy:.2f}) "
        f"half=({patch.hx:.2f},{patch.hy:.2f}) mu_d={skeleton.demo_cfg.ice.mu_d}"
    )
    fired_detail = f" ({first.summary})" if first else ""
    print(f"monitor fired: {result.monitor_fired}{fired_detail}")
    for line in skeleton.policy.transitions:
        print(f"recovery: {line}")
    print(f"shield interventions: {result.shield_interventions} (pass-through stub)")
    print(f"fall: {result.fell}")
    print(f"goal reached: {result.goal_reached} dist={result.final_dist_m:.2f}m")
    print(f"sim time: {result.sim_time_s:.1f}s steps={result.n_steps}")
    print(f"trajectory hash: {result.traj_hash}")
    budget = float(skeleton.demo_cfg.wall_clock_budget_s)
    print(f"wall clock: {result.wall_time_s:.1f}s (budget {budget:.0f}s)")

    ok = (
        result.monitor_fired
        and not result.fell
        and result.goal_reached
        and result.wall_time_s < budget
    )
    print("PASS: walking skeleton demo" if ok else "FAIL: walking skeleton demo")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
