from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_kinofail_reconfirmation_t2_f14 import (
    SCENE,
    case_recovery_state,
)


def _row(index: int) -> dict[str, str]:
    return {
        "case_id": f"case_{index:02d}",
        "scene_cluster": SCENE,
    }


def _manifest(root: Path, index: int, *, passed: bool = True) -> None:
    folder = root / f"case_{index:02d}"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": f"case_{index:02d}",
                "scene_cluster": SCENE,
                "passed": passed,
            }
        ),
        encoding="utf-8",
    )


def test_f14_partitions_sealed_zero_and_never_attempted(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(5)]
    _manifest(tmp_path, 0)
    _manifest(tmp_path, 1)
    (tmp_path / "case_02").mkdir()

    state = case_recovery_state(scene_rows=rows, out=tmp_path)

    assert state["sealed_manifest_count"] == 2
    assert state["zero_observation_case_ids"] == ["case_02"]
    assert state["never_attempted_case_ids"] == ["case_03", "case_04"]
    assert state["existing_failed_manifest_count"] == 0
    assert state["case_ids_partition_schedule"] is True


def test_f14_refuses_to_replay_existing_failed_manifest(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(3)]
    _manifest(tmp_path, 0, passed=False)
    (tmp_path / "case_01").mkdir()

    with pytest.raises(RuntimeError, match="forbids replay"):
        case_recovery_state(scene_rows=rows, out=tmp_path)


def test_f14_refuses_zero_observation_directory_with_files(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(3)]
    _manifest(tmp_path, 0)
    folder = tmp_path / "case_01"
    folder.mkdir()
    (folder / "partial.bin").write_bytes(b"observation")

    with pytest.raises(RuntimeError, match="contains artifact files"):
        case_recovery_state(scene_rows=rows, out=tmp_path)
