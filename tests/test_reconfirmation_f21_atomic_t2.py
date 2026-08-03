from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import isaac_collect_kinofail_confirmatory_t2_case_f21 as case
from scripts import recover_kinofail_reconfirmation_scene05_f21 as recovery
from scripts import run_kinofail_reconfirmation_scene_pipeline_v5 as pipeline
from scripts import run_kinofail_reconfirmation_scene_pipeline_v6 as pipeline_v6
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as runner


SCENE = "unit_scene"


def _write_valid_case(root: Path, case_id: str, *, passed: bool = True) -> Path:
    case_root = root / case_id
    case_root.mkdir(parents=True)
    observables = case_root / "observables.npz"
    observables.write_bytes(b"observables")
    image_hashes: dict[str, str] = {}
    for index in range(30):
        image = case_root / f"image_{index:02d}.png"
        image.write_bytes(f"image-{index}".encode())
        image_hashes[image.name] = runner.sha256(image)
    manifest = {
        "case_id": case_id,
        "scene_cluster": SCENE,
        "passed": passed,
        "artifacts": {
            "observables": {
                "path": observables.name,
                "sha256": runner.sha256(observables),
            },
            "image_sha256": image_hashes,
        },
    }
    path = case_root / "manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return path


def test_single_case_wrapper_selects_exact_frozen_row() -> None:
    rows = [{"case_id": "a"}, {"case_id": "b"}]
    assert case.select_case_rows(rows, case_id="b") == [{"case_id": "b"}]
    with pytest.raises(RuntimeError, match="expected one"):
        case.select_case_rows(rows, case_id="missing")


def test_corpus_state_requires_contiguous_admitted_prefix(tmp_path: Path) -> None:
    rows = [{"case_id": value} for value in ("a", "b", "c")]
    _write_valid_case(tmp_path, "a")
    (tmp_path / "b").mkdir()
    state = runner.corpus_case_state(rows=rows, out=tmp_path, scene=SCENE)
    assert state["sealed_manifest_count"] == 1
    assert state["empty_zero_observation_case_ids"] == ["b"]
    assert state["never_attempted_case_ids"] == ["c"]

    _write_valid_case(tmp_path, "c")
    with pytest.raises(RuntimeError, match="prefix"):
        runner.corpus_case_state(rows=rows, out=tmp_path, scene=SCENE)


def test_corpus_state_never_retries_admitted_failure(tmp_path: Path) -> None:
    rows = [{"case_id": "a"}]
    _write_valid_case(tmp_path, "a", passed=False)
    with pytest.raises(RuntimeError, match="never retries"):
        runner.corpus_case_state(rows=rows, out=tmp_path, scene=SCENE)


def test_atomic_commit_promotes_only_validated_complete_case(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staging" / "case_a"
    _write_valid_case(staged.parent, staged.name)
    final = tmp_path / "corpus" / staged.name
    final.parent.mkdir()
    final.mkdir()
    source_sha256 = runner.sha256(staged / "manifest.json")

    manifest = runner._commit_case(
        staged_case=staged,
        final_case=final,
        case_id=staged.name,
        scene=SCENE,
    )

    assert manifest["passed"] is True
    assert not staged.exists()
    assert runner.sha256(final / "manifest.json") == source_sha256
    assert len(list(final.glob("image_*.png"))) == 30


def test_process_group_termination_bounds_a_hang() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    action = runner._terminate_group(process)
    assert action["pid"] == process.pid
    assert action["sigterm"] is True
    assert process.poll() is not None


def test_scene_pipeline_installs_atomic_t2_entrypoint() -> None:
    target = pipeline.predecessor.predecessor
    original = target.SLOTTED_T2
    try:
        pipeline.install_f21_t2_runner()
        assert target.SLOTTED_T2 == pipeline.ATOMIC_T2
    finally:
        target.SLOTTED_T2 = original


def test_recovery_process_matching_requires_an_exact_script_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = {
        "pid": 1,
        "command": [
            "bash",
            "-c",
            "python scripts/worker.py --scene scene",
        ],
    }
    python = {
        "pid": 2,
        "command": ["python", "scripts/worker.py", "--scene", "scene"],
    }
    monkeypatch.setattr(
        recovery.sealer,
        "matching_processes",
        lambda *_args: [shell, python],
    )
    assert recovery._exact_token_processes(
        "worker.py", "--scene", "scene"
    ) == [python]


def test_f21_manifest_authenticates_all_successor_sources() -> None:
    amendment = runner._validate_f21()
    assert amendment["passed"] is True


def test_f23_validates_f17_before_installing_atomic_t2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sentinel = {"passed": True}

    def validate_f17() -> dict[str, bool]:
        events.append("validate_f17")
        return sentinel

    def install_f21() -> None:
        events.append("install_f21")

    monkeypatch.setattr(
        pipeline_v6.F17_PIPELINE, "_validate_f17", validate_f17
    )
    monkeypatch.setattr(
        pipeline_v6.f21, "install_f21_t2_runner", install_f21
    )
    assert pipeline_v6.install_f23_t2_runner() is sentinel
    assert events == ["validate_f17", "install_f21"]


def test_f23_manifest_authenticates_successor_sources() -> None:
    amendment = pipeline_v6._validate_f23()
    assert amendment["passed"] is True
