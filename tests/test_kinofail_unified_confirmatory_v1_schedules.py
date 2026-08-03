from __future__ import annotations

import json
import hashlib
from pathlib import Path

import yaml

from scripts import build_kinofail_unified_confirmatory_v1_schedules as build
from scripts.freeze_kinofail_unified_confirmatory_f0 import confirmatory_design


ROOT = Path(__file__).resolve().parents[1]


def test_f0_material_aliases_are_exact_and_domain_balanced() -> None:
    catalog = yaml.safe_load(
        (
            ROOT / "configs/data/kinofail_confirmatory_terrain_pbr_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    materials = catalog["materials"]
    assert len(materials) == 30
    for domain in ("life", "production", "wild"):
        observed = [
            row["id"] for row in materials if row["domains"] == [domain]
        ]
        assert observed == [
            f"confirm_v1_{domain}_pbr_{index:02d}" for index in range(10)
        ]
    assert len({row["source_asset_id"] for row in materials}) == 30


def test_authoritative_design_references_only_the_frozen_candidate_stream() -> None:
    design = json.loads(
        (
            ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        design["candidate_source"]["path"]
        == "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json"
    )
    assert design["candidate_source"]["life_candidates"] == 20
    assert design["candidate_source"]["production_candidates"] == 20
    assert design["candidate_source"]["wild_fixed_scenes"] == 10
    assert "first_10_model_blind_QA_passes" in design["candidate_source"][
        "selection"
    ]["life"]
    assert design["scale"]["counterfactual_pairs"] == 10_560
    assert design["conflict"]["cases_per_cell"] == 1_500


def test_interpolation_and_physical_nuisance_are_frozen() -> None:
    design = json.loads(
        (
            ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert build.interpolate_parameters(
        {"x": 2.0, "fixed": [1.0, 4.0]},
        {"x": 6.0, "fixed": [1.0, 8.0]},
        0.25,
    ) == {"fixed": [1.0, 5.0], "x": 3.0}
    nuisance = build.physical_nuisance(
        design, profile_index=7, physics_seed=2_030_000_123
    )
    assert set(nuisance) == build.NUISANCE_KEYS
    assert nuisance["profile_index"] == 7
    assert nuisance["physics_seed"] == 2_030_000_123
    assert nuisance["pair_shared"] is True


def test_scale_pair_has_explicit_audit_fields_and_shared_nuisance() -> None:
    design = json.loads(
        (
            ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
        ).read_text(encoding="utf-8")
    )
    f0 = {
        "design": {
            "scenes": [
                {
                    "scene_index": 0,
                    "scene_id": "confirm_v1_life_scene_00",
                }
            ]
        }
    }
    scene = {
        "scene_id": "confirm_v1_life_scene_00",
        "source_scene_id": "confirm_life_001",
        "source": "fixture",
        "source_scene_seed": 2027012801,
        "runtime_seed": 2027012802,
        "domain": "life",
        "material_ids": [
            "confirm_v1_life_pbr_00",
            "confirm_v1_life_pbr_01",
        ],
        "geometry_hash": "b" * 64,
        "compiled_audit_sha256": "a" * 64,
    }
    materials = {
        material_id: {
            "id": material_id,
            "source_asset_id": source,
            "semantic_family": source.lower(),
            "source": f"https://ambientcg.com/a/{source}",
            "license": "CC0",
            "physical_size_m": [2.0, 2.0],
        }
        for material_id, source in (
            ("confirm_v1_life_pbr_00", "Tiles144"),
            ("confirm_v1_life_pbr_01", "Tiles143"),
        )
    }
    operator = {
        "id": "O1_mu_field",
        "category": "low_friction",
        "nominal": {"mu_s": 0.8, "mu_d": 0.6},
        "M": {"mu_s": 0.2, "mu_d": 0.15},
        "S": {"mu_s": 0.1, "mu_d": 0.07},
        "physical_realization": "physx_region_material",
        "geometry_profile": "irregular_material_region",
        "counterfactual_inherit": [],
    }
    rows = build._scale_context_rows(
        design=design,
        f0=f0,
        scene=scene,
        context={
            "context_index": 0,
            "scene_id": scene["scene_id"],
            "material_id": "confirm_v1_life_pbr_00",
            "material_slot": 0,
        },
        materials=materials,
        operators=[operator],
    )
    assert len(rows) == 32
    first_pair = rows[:2]
    assert {row["condition"] for row in first_pair} == {
        "nominal_counterfactual",
        "anomaly",
    }
    assert len({row["counterfactual_group_id"] for row in first_pair}) == 1
    required = {
        "scene_index",
        "scene_id",
        "material_slot",
        "cluster_material",
        "operator_index",
        "replicate_index",
        "operator_seed",
        "physical_nuisance",
    }
    assert all(required <= set(row) for row in first_pair)
    assert first_pair[0]["physical_nuisance"] == first_pair[1][
        "physical_nuisance"
    ]
    assert first_pair[0]["operator_seed"] == first_pair[0][
        "physical_nuisance"
    ]["physics_seed"]


def test_conflict_case_expands_to_six_explicit_model_records() -> None:
    case = {
        "case_id": "case_1",
        "case_seed": 2_040_000_001,
        "scene_index": 0,
        "scene_id": "confirm_v1_life_scene_00",
        "scene_cluster": "confirm_v1_life_scene_00",
        "scene_family": "confirm_v1_life_scene_00",
        "domain": "life",
        "material_id": "confirm_v1_life_pbr_00",
        "material_slot": 0,
        "local_case_index": 1,
        "physical_nuisance": {
            "profile_index": 1,
            "start_progress_m": 0.01,
            "start_lateral_offset_m": 0.0,
            "start_heading_offset_rad": 0.0,
            "forward_speed_mps": 0.24,
            "controller_target_lateral_offset_m": 0.0,
            "physics_seed": 2_040_000_001,
            "pair_shared": True,
        },
    }
    rows = build._conflict_model_rows(
        case,
        cell="T2_vision_decisive",
        candidates=("O2_compliance", "O4_tether"),
    )
    assert len(rows) == 6
    required = {
        "cell",
        "case_id",
        "case_seed",
        "scene_index",
        "scene_id",
        "material_slot",
        "cluster_material",
        "local_case_index",
        "record_role",
        "cause_id",
        "view_id",
    }
    assert all(required <= set(row) for row in rows)
    assert {row["cause_id"] for row in rows} == {
        "O2_compliance",
        "O4_tether",
    }
    assert {row["view_id"] for row in rows} == {
        "primary",
        "swap_01",
        "swap_02",
    }


def test_collection_protocol_satisfies_runtime_scope_contract(tmp_path) -> None:
    schedule_stage = tmp_path / "schedule.jsonl"
    row = {
        "scene_family": "confirm_v1_life_scene_00",
        "counterfactual_group_id": "cf_fixture",
        "target_operator": "O1_mu_field",
        "condition": "anomaly",
        "physical_realization": "physx_region_material",
        "geometry_profile": "irregular_material_region",
        "severity_id": "moderate",
    }
    schedule_stage.write_text(
        json.dumps(row) + "\n" + json.dumps({**row, "condition": "nominal_counterfactual"}) + "\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "fixture"
    fixture.write_text("fixture", encoding="utf-8")
    protocol = build._collection_protocol(
        protocol_id="fixture",
        schedule_final_path=tmp_path / "final.jsonl",
        schedule_stage_path=schedule_stage,
        registry_path=fixture,
        lock_path=fixture,
        collector_path=fixture,
        runtime_manifest_path=fixture,
        f0_path=fixture,
        design_path=fixture,
        scene_id="confirm_v1_life_scene_00",
        pairs=1,
        operators=["O1_mu_field"],
        root=tmp_path,
    )
    assert protocol["status"] == "frozen"
    assert protocol["benchmark_id"] == "kinofail_unified_confirmatory_v1"
    assert protocol["allowed"]["counterfactual_group_ids"] == ["cf_fixture"]
    assert protocol["allowed"]["physical_realizations"] == [
        "physx_region_material"
    ]
    assert protocol["allowed"]["geometry_profiles"] == [
        "irregular_material_region"
    ]


def test_full_schedule_compiler_realizes_the_f0_ledger(tmp_path) -> None:
    internal = confirmatory_design()
    candidate_config = json.loads(
        (
            ROOT
            / "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json"
        ).read_text(encoding="utf-8")
    )
    selected = {
        domain: [
            row
            for row in (
                candidate_config["wild_scenes"]
                if domain == "wild"
                else candidate_config["indoor_candidates"]
            )
            if row["domain"] == domain
        ][:10]
        for domain in ("life", "production", "wild")
    }
    assignments = {
        scene["scene_id"]: [
            row["material_id"]
            for row in sorted(
                (
                    item
                    for item in internal["scene_material_assignments"]
                    if item["scene_id"] == scene["scene_id"]
                ),
                key=lambda item: item["material_slot"],
            )
        ]
        for scene in internal["scenes"]
    }
    counters = {"life": 0, "production": 0, "wild": 0}
    scenes = []
    for scene in internal["scenes"]:
        domain = scene["domain"]
        candidate = selected[domain][counters[domain]]
        counters[domain] += 1
        row = {
            "scene_id": scene["scene_id"],
            "candidate_id": candidate["scene_id"],
            "source_scene_id": candidate["scene_id"],
            "source": "fixture",
            "domain": domain,
            "runtime_seed": candidate["runtime_seed"],
            "geometry_hash": hashlib.sha256(
                f"geometry-{scene['scene_id']}".encode()
            ).hexdigest(),
            "material_ids": assignments[scene["scene_id"]],
            "selected_by_model_blind_prefix_rule": True,
            "terminal_scene_admission_sha256": hashlib.sha256(
                f"terminal-{scene['scene_id']}".encode()
            ).hexdigest(),
        }
        row[
            "metric_geometry_seed" if domain == "wild" else "source_scene_seed"
        ] = candidate.get("metric_geometry_seed", candidate.get("source_seed"))
        scenes.append(row)
    registry = {
        "selection_uses_model_predictions": False,
        "scenes": scenes,
        "attrition": {
            domain: {
                "attempts": [
                    {
                        "candidate_index": index,
                        "candidate": candidate,
                        "terminal_audit_sha256": hashlib.sha256(
                            f"terminal-{candidate['scene_id']}".encode()
                        ).hexdigest(),
                        "admitted": True,
                    }
                    for index, candidate in enumerate(selected[domain])
                ]
            }
            for domain in ("life", "production", "wild")
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    catalog = yaml.safe_load(
        (
            ROOT / "configs/data/kinofail_confirmatory_terrain_pbr_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    materials = []
    for index, source in enumerate(catalog["materials"]):
        row = dict(source)
        row["archive_sha256"] = hashlib.sha256(
            f"archive-{index}".encode()
        ).hexdigest()
        row["maps"] = {
            name: {
                "sha256": hashlib.sha256(
                    f"map-{index}-{name}".encode()
                ).hexdigest(),
                "bytes": 1,
            }
            for name in ("basecolor", "normal", "roughness")
        }
        materials.append(row)
    lock_path = tmp_path / "material_lock.json"
    lock_path.write_text(json.dumps({"materials": materials}), encoding="utf-8")

    design_path = (
        ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
    )
    bound_paths = [
        build.COLLECTOR_PATH,
        build.RUNTIME_MANIFEST_PATH,
        build.T2_COLLECTOR_PATH,
        build.T2_VISUAL_HELPER_PATH,
        Path(build.__file__).resolve(),
    ]
    f0 = {
        "status": "frozen_before_new_scene_generation",
        "external_design": {
            "sha256": hashlib.sha256(design_path.read_bytes()).hexdigest()
        },
        "design": internal,
        "frozen_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in bound_paths
        ],
    }
    f0_path = tmp_path / "f0.json"
    f0_path.write_text(json.dumps(f0), encoding="utf-8")
    output = tmp_path / "schedules"
    result = build.build_schedules(
        design_path=design_path,
        f0_path=f0_path,
        registry_path=registry_path,
        lock_path=lock_path,
        collector_path=build.COLLECTOR_PATH,
        runtime_manifest_path=build.RUNTIME_MANIFEST_PATH,
        output_root=output,
    )
    assert result["counts"]["scale_counterfactual_pairs"] == 10_560
    assert result["counts"]["t2_cases"] == 1_500
    assert result["counts"]["t3_cases"] == 1_500
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["passed"] is True
    assert (output / "c2_t2/collection_protocol.json").is_file()
    assert (output / "c2_t3/audit.json").is_file()
