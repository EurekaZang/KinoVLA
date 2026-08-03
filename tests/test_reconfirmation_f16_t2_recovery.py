from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_kinofail_reconfirmation_t2_f16 import (
    EXPECTED_CASE_COUNT,
    EXPECTED_SEALED_PREFIX,
    SCENE,
    case_recovery_state,
)


def _rows() -> list[dict[str, str]]:
    return [
        {"case_id": f"case_{index:02d}", "scene_cluster": SCENE}
        for index in range(EXPECTED_CASE_COUNT)
    ]


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


def test_f16_partitions_sealed_zero_and_never_attempted(
    tmp_path: Path,
) -> None:
    rows = _rows()
    for index in range(EXPECTED_SEALED_PREFIX):
        _manifest(tmp_path, index)
    (tmp_path / f"case_{EXPECTED_SEALED_PREFIX:02d}").mkdir()

    state = case_recovery_state(rows=rows, out=tmp_path)

    assert state["sealed_manifest_count"] == EXPECTED_SEALED_PREFIX
    assert state["zero_observation_case_ids"] == ["case_30"]
    assert len(state["never_attempted_case_ids"]) == 19
    assert state["existing_failed_manifest_count"] == 0
    assert state["case_ids_partition_schedule"] is True


def test_f16_refuses_existing_failed_manifest(tmp_path: Path) -> None:
    rows = _rows()
    for index in range(EXPECTED_SEALED_PREFIX - 1):
        _manifest(tmp_path, index)
    _manifest(tmp_path, EXPECTED_SEALED_PREFIX - 1, passed=False)
    (tmp_path / f"case_{EXPECTED_SEALED_PREFIX:02d}").mkdir()

    with pytest.raises(RuntimeError, match="forbids replay"):
        case_recovery_state(rows=rows, out=tmp_path)


def test_f16_refuses_zero_observation_directory_with_files(
    tmp_path: Path,
) -> None:
    rows = _rows()
    for index in range(EXPECTED_SEALED_PREFIX):
        _manifest(tmp_path, index)
    folder = tmp_path / f"case_{EXPECTED_SEALED_PREFIX:02d}"
    folder.mkdir()
    (folder / "partial.bin").write_bytes(b"observation")

    with pytest.raises(RuntimeError, match="contains artifact files"):
        case_recovery_state(rows=rows, out=tmp_path)
