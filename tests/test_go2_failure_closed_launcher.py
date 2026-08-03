from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_embodiedgen_go2_scene_qa_failure_closed_v1.py"
SPEC = importlib.util.spec_from_file_location("go2_failure_closed_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GO2_SCRIPT = MODULE.GO2_SCRIPT
ISAAC_SETUP = MODULE.ISAAC_SETUP
ISAACLAB_PYTHON = MODULE.ISAACLAB_PYTHON
SHELL = MODULE.SHELL
launcher_components = MODULE.launcher_components


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_components_bind_the_only_supported_runtime() -> None:
    records = launcher_components()
    expected = [SHELL, ISAAC_SETUP, ISAACLAB_PYTHON, GO2_SCRIPT]
    assert [item["path"] for item in records] == [str(path.resolve()) for path in expected]
    assert [item["sha256"] for item in records] == [
        _sha256(path.resolve()) for path in expected
    ]


def test_launcher_uses_kinovla_conda_python_not_isaacsim_python() -> None:
    assert ISAACLAB_PYTHON == Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
    assert ISAAC_SETUP == Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
    assert "isaac_embodiedgen_go2_scene_qa.py" == GO2_SCRIPT.name
