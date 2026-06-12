"""GPU-only bring-up gate: runs scripts/stand_go2.py headless (M0 exit criterion).

Auto-skipped on machines without Isaac Lab (see conftest.py). On the GPU machine:
    pytest -m sim tests/test_sim_bringup.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.sim
@pytest.mark.slow
def test_go2_stands_headless():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "stand_go2.py"), "--headless"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stand_go2.py failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS: Go2 standing check" in result.stdout
