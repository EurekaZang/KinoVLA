from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_o4_scene_candidate_v30 as runner


def test_usd_command_explicitly_restores_kit_pxr_path() -> None:
    command = runner._usd_command(Path("/tmp/audit.py"), ["--out", "/tmp/a.json"])
    assert command[:2] == [str(runner.implementation.SHELL), "-lc"]
    assert str(runner.implementation.ISAAC_SETUP) in command
    assert str(runner.KIT_SITE_PACKAGES) in command
    assert command[-3:] == ["/tmp/audit.py", "--out", "/tmp/a.json"]


def test_v30_usd_stage_uses_corrected_command(monkeypatch) -> None:
    captured = {}

    def fake_original(**kwargs):
        captured.update(kwargs)
        return True, {}

    monkeypatch.setattr(runner.predecessor, "_original_run_stage", fake_original)
    runner._run_stage_v30(
        name="source_preflight",
        command=["python", "/tmp/audit.py"],
        audit_path=Path("/tmp/audit.json"),
        environment={},
    )
    assert str(runner.KIT_SITE_PACKAGES) in captured["command"]
