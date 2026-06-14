"""GPU-gated sim checks (manual gate per QA 5.1.3 — run on the GPU machine).

The Isaac walking-skeleton demo is the M1/M2 sim gate: it exercises scene bring-up,
the O1 PhysX material patch with μ readback, and the full monitor→FSM→shield loop on
the Go2 asset. At M2 the Go2 is physically simulated and walked by the trained RSL-RL
policy (kino_vla/sim/isaac_policy_backend.py), and slip is measured from real foot
contact — so this gate also certifies the locomotion policy integration.

Run with:  pytest -m sim
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kino_vla.utils.config import REPO_ROOT


@pytest.mark.sim
@pytest.mark.slow
def test_walking_skeleton_demo_isaac():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_demo.py"),
            "--backend",
            "isaac",
            "--headless",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,  # first launch compiles shaders / downloads assets
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # The trained policy walks the Go2 (M2 locomotion integration).
    assert "policy backend" in out
    # θ-application (QA 5.2a): the μ PhysX actually holds matches the set value.
    assert "O1 patch material applied" in out
    assert "monitor fired: True" in out
    assert "fall: False" in out
    assert "goal reached: True" in out
    assert "PASS: walking skeleton demo" in out
