from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/eval/kinofail_realistic_a7_matched_design_v1.json"
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a7_matched_subsets_are_complete_predeclared_scene_blocks() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in rows
        if row["condition"] == "anomaly"
    }
    seeds_by_scene: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        seeds_by_scene[str(row["scene_family"])].add(int(row["scene_seed"]))
    profile = {
        (scene, seed): index
        for scene, seeds in seeds_by_scene.items()
        for index, seed in enumerate(sorted(seeds))
    }

    visual_ids = design["design"]["visual_camera_replay"]["counterfactual_group_ids"]
    visual_rows = [anomaly[pair_id] for pair_id in visual_ids]
    assert len(visual_rows) == 9
    assert len({row["scene_family"] for row in visual_rows}) == 9
    assert {row["target_operator"] for row in visual_rows} == {"O5_payload"}
    assert {row["severity_id"] for row in visual_rows} == {"moderate"}
    assert {
        profile[(row["scene_family"], int(row["scene_seed"]))]
        for row in visual_rows
    } == {2}

    physics_ids = design["design"]["local_vs_surrogate_physics"][
        "counterfactual_group_ids"
    ]
    physics_rows = [anomaly[pair_id] for pair_id in physics_ids]
    assert len(physics_rows) == 18
    assert {
        (row["scene_family"], row["target_operator"])
        for row in physics_rows
    } == {
        (scene, operator)
        for scene in seeds_by_scene
        for operator in ("O2_compliance", "O4_tether")
    }
    assert {row["severity_id"] for row in physics_rows} == {"moderate"}
    assert {
        profile[(row["scene_family"], int(row["scene_seed"]))]
        for row in physics_rows
    } == {2}
    assert design["a8_in_scope"] is False
    assert design["model_outcomes_available_at_freeze"] is False


def test_a7_derived_collection_schedules_and_protocols_are_hash_frozen() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    cases = {
        "legacy": {
            "schedule": ROOT
            / "outputs/kinofail_realistic/design_a7_legacy_surrogate_v1/schedule.jsonl",
            "protocol": ROOT
            / "configs/data/kinofail_realistic_a7_legacy_surrogate_formal_v1.json",
            "collector": ROOT
            / "scripts/isaac_collect_kinofail_realistic_a7_legacy_surrogate_v1.py",
            "pair_ids": set(
                design["design"]["local_vs_surrogate_physics"][
                    "counterfactual_group_ids"
                ]
            ),
            "operators": {"O2_compliance", "O4_tether"},
            "pairs": 18,
        },
        "visual": {
            "schedule": ROOT
            / "outputs/kinofail_realistic/design_a7_visual_replay_v1/schedule.jsonl",
            "protocol": ROOT
            / "configs/data/kinofail_realistic_a7_visual_replay_formal_v1.json",
            "collector": ROOT
            / "scripts/isaac_collect_kinofail_realistic_a7_visual_replay_v1.py",
            "pair_ids": set(
                design["design"]["visual_camera_replay"][
                    "counterfactual_group_ids"
                ]
            ),
            "operators": {"O5_payload"},
            "pairs": 9,
        },
    }
    for case in cases.values():
        protocol = json.loads(case["protocol"].read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in case["schedule"].read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(rows) == 2 * case["pairs"]
        assert {row["counterfactual_group_id"] for row in rows} == case["pair_ids"]
        assert {row["target_operator"] for row in rows} == case["operators"]
        assert {row["physical_nuisance_profile_index"] for row in rows} == {2}
        assert len({row["scene_family"] for row in rows}) == 9
        assert protocol["schedule_sha256"] == _sha256(case["schedule"])
        assert protocol["collector_sha256"] == _sha256(case["collector"])
        assert protocol["a7_design_sha256"] == _sha256(DESIGN)
        assert protocol["collection_contract"]["a8_in_scope"] is False
        assert protocol["collection_contract"][
            "model_outcomes_available_at_freeze"
        ] is False


def test_a7_snapshot_and_analysis_protocols_bind_their_implementations() -> None:
    snapshot = json.loads(
        (
            ROOT / "configs/eval/kinofail_realistic_a7_matched_snapshot_v1.json"
        ).read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (
            ROOT / "configs/eval/kinofail_realistic_a7_matched_analysis_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert snapshot["input_sha256"]["builder"] == _sha256(
        ROOT / "scripts/build_kinofail_realistic_a7_matched_snapshots_v1.py"
    )
    assert snapshot["input_sha256"]["design"] == _sha256(DESIGN)
    assert analysis["input_sha256"]["analyzer"] == _sha256(
        ROOT / "scripts/analyze_kinofail_realistic_a7_matched_v1.py"
    )
    assert analysis["input_sha256"]["snapshot_protocol"] == _sha256(
        ROOT / "configs/eval/kinofail_realistic_a7_matched_snapshot_v1.json"
    )
    assert analysis["training_seeds"] == [0, 1, 2]
    assert analysis["a8_in_scope"] is False


def test_scale_v8_frozen_runtime_dependencies_remain_byte_identical() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["collector_sha256"] == _sha256(
        ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
    )
    assert protocol["runtime_manifest_sha256"] == _sha256(
        ROOT / "kino_vla/data/runtime_manifest.py"
    )
