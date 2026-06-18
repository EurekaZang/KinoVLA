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
    # M5: the semantic traversability map ran on the physically-simulated Go2 — the real
    # slip overwrote costmap cells and the map fed avoid discs to the planner (spec §7).
    assert "semantic map: physical_cells=" in out
    map_line = next(line for line in out.splitlines() if "semantic map:" in line)
    n_phys = int(map_line.split("physical_cells=")[1].split()[0])
    assert n_phys > 0, f"map should have overwritten cells from the Isaac slip: {map_line}"
    assert "PASS: walking skeleton demo" in out


@pytest.mark.sim
@pytest.mark.slow
def test_m5_operators_isaac():
    """M5 O2/O4/O7 on the physically-simulated Go2 (spec §8.2; CLAUDE.md M5 GPU gate)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_m5_check.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: M5 Isaac operator gate" in proc.stdout


@pytest.mark.sim
@pytest.mark.slow
def test_m2_operators_isaac():
    """M2 O3/O5/O8/O9/O10 on the physically-simulated Go2 (spec §8.2; GPU backfill).

    Each operator runs in its own lateral lane so a topple on one cannot starve the
    others; asserts the real PhysX effects (μ-swap+slip, mass↑, wall-block, foot-unload,
    effort-starve)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_m2_ops_check.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: M2 Isaac operator gate" in proc.stdout


@pytest.mark.sim
@pytest.mark.slow
def test_m3_cbf_adversarial_isaac():
    """M3 CBF shield actively clamps hostile commands on the real Go2 (spec §6.6).

    Honest bar (deviations #9/#13): the trained policy is command-robust and the CBF's
    zero-fall guarantee is a reduced-LIP-model property (the fall-count contrast lives in
    the surrogate gate). On Isaac the demonstrable claim is that the shield intervenes on
    most steps and bounds the issued command below the hostile input."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_cbf_adversarial.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: shield actively clamps hostile commands" in proc.stdout


@pytest.mark.sim
@pytest.mark.slow
def test_m4_kino_tokens_isaac():
    """M4 Kino-Tokens extractor trained on REAL Go2 proprioception (spec §4, §6.5).

    Drives the physically-simulated Go2 across firm/ice lanes, trains the extractor on the
    collected windows, and verifies μ̂ separates ice from firm and the μ̂→CBF-shield coupling
    tightens the friction cone on detected ice (the spec §6.5 payoff on the real robot)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_tokens_check.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: M4 Kino-Tokens on Isaac" in proc.stdout


@pytest.mark.sim
@pytest.mark.slow
def test_m3_cbf_pushfall_isaac():
    """M3 push-fall characterization on the real Go2 (spec §6.6; CLAUDE.md §6 #22).

    Turns the previously-untested 0/0 zero-fall contrast into a real 36-scenario sweep: a real
    O6 impulse builds a capture-point regime, then a hostile command is run shielded vs
    bypassed. FINDING (deviations #9/#13): the reduced-LIP CBF never reduces falls on the
    command-robust full-order policy — for forward/diagonal pushes it is anti-protective. The
    guarantee is a reduced-LIP-model property; the surrogate gate stays the falsifiable
    zero-fall test. The run succeeds once it has characterized the effect."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_cbf_pushfall.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: M3 push-fall characterization complete" in proc.stdout


@pytest.mark.sim
@pytest.mark.slow
def test_m6_hindsight_isaac():
    """M6 Hindsight-CoT pipeline on the REAL Go2 (spec §10; the data pipeline is Isaac-based,
    §1/§8.1). Drives the physically-simulated Go2 into each operator's failure (lateral lanes),
    intercepts with the collection monitor, and snapshots the REAL proprioception + privileged
    θ — then runs the SAME Oracle + truth-consistency filter. Verifies all lanes intercept,
    each snapshot's real θ confirms its operator (ice μ≈0.1, payload>0, effort<0.5), and the
    filter keeps grounded reflections (closes the M6 surrogate gap on GPU, like M4/M5 #21/#23)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_hindsight_check.py"), "--headless"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: M6 Hindsight on Isaac" in proc.stdout
