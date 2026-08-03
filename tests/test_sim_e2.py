"""GPU-only gate #14: E2 proprioceptive-ceiling A/B dichotomy on the real Go2.

Runs scripts/isaac_e2_check.py --headless --quick: asserts the full comparison (B1/B2/B5 ×
Suite-Sem attribution + recovery + Suite-Cal) runs end-to-end on the real stack and the verdict
holds (B5≫B1 on Suite-Sem attribution; B1≈B5 on Suite-Cal parity). The G1/G2 trapping-matched
calibration runs inside; the result card (incl. the fallback case) lands in outputs/eval/e2/.

Auto-skipped without Isaac Lab (conftest.py). On the GPU box: pytest -m sim tests/test_sim_e2.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.sim
@pytest.mark.slow
def test_e2_dichotomy():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "isaac_e2_check.py"), "--headless", "--quick"],
        capture_output=True,
        text=True,
        timeout=5400,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"isaac_e2_check.py failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS: E2 proprioceptive-ceiling dichotomy" in result.stdout
