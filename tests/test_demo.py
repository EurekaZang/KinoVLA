"""Walking-skeleton demo gates (M1 exit criteria + QA 5.1.4 permanent regression).

Covers: end-to-end assertions on the surrogate backend, seeded determinism,
non-vacuity (a non-recovering policy falls on the same episode), and the pinned
subprocess output contract of scripts/run_demo.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from kino_vla.loop import run_episode
from kino_vla.skeleton import build_walking_skeleton, run_walking_skeleton
from kino_vla.utils.config import REPO_ROOT, load_config

DEMO_SEED = 42  # pinned demo seed (configs/default.yaml)


def test_demo_episode_end_to_end():
    result, skeleton = run_walking_skeleton(DEMO_SEED)
    assert result.monitor_fired, "monitor must fire on the ice patch"
    assert not result.fell, "robot must not fall"
    assert result.goal_reached, "robot must reach the goal"
    # M3: the CBF-QP shield (spec §6) replaced the pass-through stub. It is transparent
    # at the FSM cruise speed and only bites hard-decel/backstep transients, so it must
    # not dominate the episode (goal_reached above already rules out a strangling shield,
    # e.g. an over-aggressive K_ξ that throttles the cruise command).
    assert 0 <= result.shield_interventions < result.n_steps // 2
    assert skeleton.policy.backstep_count >= 1
    assert result.events[0].channel == "slip_ratio"
    # The event fired on the patch, not on nominal ground.
    assert skeleton.terrain.hazard_patch.contains(result.events[0].pos)


def test_demo_deterministic_same_seed():
    r1, _ = run_walking_skeleton(DEMO_SEED)
    r2, _ = run_walking_skeleton(DEMO_SEED)
    assert r1.traj_hash == r2.traj_hash
    assert r1.n_steps == r2.n_steps


def test_demo_seed_changes_trajectory():
    r1, _ = run_walking_skeleton(DEMO_SEED)
    r2, _ = run_walking_skeleton(DEMO_SEED + 1)
    assert r1.traj_hash != r2.traj_hash


class GoStraightPolicy:
    """Non-recovering policy: full cruise at the goal, ignores monitor events."""

    def __init__(self, goal_xy: np.ndarray, speed: float) -> None:
        self._goal = goal_xy
        self._speed = speed

    def on_event(self, event) -> bool:
        return False

    def step(self, obs) -> np.ndarray:
        to_goal = self._goal - obs.pos
        heading_err = np.arctan2(to_goal[1], to_goal[0]) - obs.heading
        return np.array([self._speed, 0.0, 2.0 * heading_err])


def test_demo_assertions_not_vacuous_without_recovery():
    # Same seed, same world, but a policy that barrels across the ice: it must
    # fall — proving the demo's "did not fall" assertion is earned by recovery.
    skeleton = build_walking_skeleton(DEMO_SEED)
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    cruise = float(load_config("recovery/fsm_v0.yaml").cruise_speed_mps)
    policy = GoStraightPolicy(goal, speed=cruise)
    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        skeleton.monitor,
        policy,
        skeleton.shield,
        seed=DEMO_SEED,
        goal_xy=goal,
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
    )
    assert result.monitor_fired  # the monitor still sees the slip
    assert result.fell
    assert not result.goal_reached


def test_run_demo_script_output_contract():
    # The pinned demo command (CLAUDE.md §1): asserted here and in CI.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_demo.py"), "--backend", "surrogate"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "monitor fired: True" in out
    assert "fall: False" in out
    assert "goal reached: True" in out
    assert "PASS: walking skeleton demo" in out


def test_repo_has_pinned_demo_entrypoint():
    assert (Path(REPO_ROOT) / "scripts" / "run_demo.py").exists()
