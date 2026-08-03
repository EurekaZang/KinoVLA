"""GPU-only gate #13: E1 proprioceptive-indistinguishability C2ST on the real Go2.

Runs scripts/isaac_e1_check.py headless in the infrastructure+controls mode (--quick, reduced
seeds): asserts the C2ST battery runs end-to-end on the real stack and the POWER control is
distinguishable (the test has power) and the vision disambiguator separates the O4↔O2 appearances.
The per-pair proprioceptive indistinguishability is reported in outputs/eval/e1/e1_results.md and
is driven by the calibration loop (run with --require-indistinguishable to certify the claim).

Auto-skipped on machines without Isaac Lab (see conftest.py). On the GPU machine:
    pytest -m sim tests/test_sim_e1.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.sim
@pytest.mark.slow
def test_e1_indistinguishability_infra():
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "isaac_e1_check.py"),
            "--headless", "--quick", "--seeds-train", "3", "--seeds-test", "2",
        ],
        capture_output=True,
        text=True,
        timeout=2400,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"isaac_e1_check.py failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS: E1 proprioceptive indistinguishability" in result.stdout
