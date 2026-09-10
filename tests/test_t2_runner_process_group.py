from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts.run_kinofail_reconfirmation_t2_scene_v2 import _launch_once


def test_t2_wrapper_exit_reclaims_lingering_descendant(
    tmp_path: Path,
) -> None:
    child_pid = tmp_path / "child.pid"
    log = tmp_path / "collector.log"
    parent = (
        "import pathlib,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    )
    result = _launch_once(
        [sys.executable, "-c", parent, str(child_pid)],
        environment=dict(os.environ),
        log_path=log,
        timeout_s=10.0,
    )
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    pid = int(child_pid.read_text())
    assert not Path(f"/proc/{pid}").exists()


def test_t2_timeout_reclaims_whole_process_group(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    log = tmp_path / "collector.log"
    parent = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    result = _launch_once(
        [sys.executable, "-c", parent, str(child_pid)],
        environment=dict(os.environ),
        log_path=log,
        timeout_s=0.5,
    )
    assert result["returncode"] == 124
    assert result["timed_out"] is True
    pid = int(child_pid.read_text())
    assert not Path(f"/proc/{pid}").exists()
