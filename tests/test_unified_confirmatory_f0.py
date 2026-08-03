from __future__ import annotations

from types import SimpleNamespace

import pytest

from kino_vla.eval.unified_moe import UnifiedEvidenceMoE
from scripts.freeze_kinofail_unified_confirmatory_f0 import (
    _design_frozen_entries,
    _fail_if_any_exists,
    _route_contract_checks,
    confirmatory_design,
    design_frozen_files,
)
from scripts.run_kinofail_confirmatory_indoor_scene_v1 import (
    _admission_decision,
)


def test_confirmatory_design_has_frozen_publication_counts() -> None:
    design = confirmatory_design()
    assert len(design["scenes"]) == 30
    assert len(design["materials"]) == 30
    assert len(design["scene_material_assignments"]) == 60
    assert len({row["scene_id"] for row in design["scenes"]}) == 30
    assert len({row["material_id"] for row in design["materials"]}) == 30
    assert design["scale"]["counterfactual_pairs"] == 10_560
    assert design["scale"]["physical_episodes"] == 21_120
    assert design["scale"]["fresh_seeds_per_scene_material_operator"] == 16
    assert design["conflict"]["cases_per_cell"] == 1_500
    assert design["conflict"]["cases_total"] == 3_000


def test_route_contract_is_strict_and_vision_wins_equal_excess() -> None:
    model = SimpleNamespace(
        classes=[f"class_{index}" for index in range(11)],
        _select_routes=UnifiedEvidenceMoE._select_routes,
    )
    checks = _route_contract_checks(model)
    assert checks == {
        "score_equal_0_8_keeps_proprio": True,
        "score_strictly_above_0_8_selects_vision": True,
        "equal_vision_joint_excess_selects_vision": True,
    }


def test_indoor_admission_distinguishes_identity_manifest_from_audits() -> None:
    names = {
        "source_manifest",
        "source_preflight",
        "base_audit",
        "corridor_audit",
        "surface_audit",
        "terrain_audit",
        "terrain_offline_audit",
        "rtx_audit",
        "go2_audit",
    }
    evidence = {
        name: {"passed": None if name == "source_manifest" else True}
        for name in names
    }
    assert _admission_decision(
        failure_stage=None,
        evidence=evidence,
        expected_names=names,
    )
    evidence["go2_audit"]["passed"] = False
    assert not _admission_decision(
        failure_stage=None,
        evidence=evidence,
        expected_names=names,
    )


def test_f0_refuses_any_existing_confirmatory_path(tmp_path) -> None:
    missing = tmp_path / "missing"
    _fail_if_any_exists([missing])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _fail_if_any_exists([missing, existing])


def test_design_frozen_file_registry_is_recursive_and_fail_closed(
    tmp_path,
) -> None:
    one = tmp_path / "one.py"
    two = tmp_path / "two.json"
    one.write_text("one\n", encoding="utf-8")
    two.write_text("{}\n", encoding="utf-8")
    entries = _design_frozen_entries(
        {
            "generation": [
                {"path": str(one)},
                {"nested": [{"path": str(two)}]},
            ]
        }
    )
    assert entries == [(str(one), None), (str(two), None)]

    design = tmp_path / "design.json"
    design.write_text(
        '{"schema_version":"test","frozen_files":'
        f'{{"nested":[{{"path":"{one}"}},{{"path":"{two}"}}]}}'
        "}\n",
        encoding="utf-8",
    )
    _value, paths = design_frozen_files(design)
    assert paths == [one.resolve(), two.resolve()]

    two.unlink()
    with pytest.raises(FileNotFoundError):
        design_frozen_files(design)
