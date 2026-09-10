from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts.run_kinofail_reconfirmation_pair_partition_v2 import (
    _run_collector_once,
)


def test_wrapper_exit_reclaims_lingering_descendant(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    log = tmp_path / "collector.log"
    parent = (
        "import pathlib,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    )
    returncode, reason, _ = _run_collector_once(
        [sys.executable, "-c", parent, str(child_pid)],
        environment=dict(os.environ),
        log_path=log,
    )
    assert returncode == 0
    assert reason is None
    pid = int(child_pid.read_text())
    assert not Path(f"/proc/{pid}").exists()
