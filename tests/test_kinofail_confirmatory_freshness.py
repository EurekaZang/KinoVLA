from __future__ import annotations

import hashlib

from scripts.audit_kinofail_confirmatory_freshness_v1 import (
    _collect_exposure,
    audit_freshness,
)
from scripts.freeze_kinofail_unified_confirmatory_f0 import (
    confirmatory_design,
)


def _empty_prior() -> dict[str, set]:
    return {
        "scenes": set(),
        "materials": set(),
        "seeds": set(),
        "content_hashes": set(),
        "operator_vectors": set(),
    }


def _realized_fixture() -> tuple[dict, dict, dict, list[dict], list[dict]]:
    design = confirmatory_design()
    f0 = {"design": design}
    candidate_by_domain = {
        domain: list(design["scene_candidate_streams"][domain])[:10]
        for domain in ("life", "production", "wild")
    }
    domain_offsets = {"life": 0, "production": 0, "wild": 0}
    realized_scenes = []
    attrition = {}
    assignments_by_scene = {
        row["scene_id"]: [
            item["material_id"]
            for item in sorted(
                (
                    candidate
                    for candidate in design["scene_material_assignments"]
                    if candidate["scene_id"] == row["scene_id"]
                ),
                key=lambda candidate: candidate["material_slot"],
            )
        ]
        for row in design["scenes"]
    }
    for domain in ("life", "production", "wild"):
        attrition[domain] = {
            "attempts": [
                {
                    "candidate_index": index,
                    "candidate": candidate,
                    "terminal_audit": f"/tmp/{candidate['scene_id']}.json",
                    "terminal_audit_sha256": hashlib.sha256(
                        f"terminal-{candidate['scene_id']}".encode()
                    ).hexdigest(),
                    "admitted": True,
                    "failure_stage": None,
                }
                for index, candidate in enumerate(candidate_by_domain[domain])
            ]
        }
    for row in design["scenes"]:
        domain = row["domain"]
        candidate = candidate_by_domain[domain][domain_offsets[domain]]
        domain_offsets[domain] += 1
        realized_scenes.append(
            {
                **row,
                "candidate_id": candidate["scene_id"],
                "source_scene_id": candidate["scene_id"],
                (
                    "metric_geometry_seed"
                    if domain == "wild"
                    else "source_scene_seed"
                ): (
                    candidate["metric_geometry_seed"]
                    if domain == "wild"
                    else candidate["source_seed"]
                ),
                "episode_sha256": hashlib.sha256(
                    row["scene_id"].encode()
                ).hexdigest(),
                "geometry_hash": hashlib.sha256(
                    f"geometry-{row['scene_id']}".encode()
                ).hexdigest(),
                "terminal_scene_admission_sha256": hashlib.sha256(
                    f"terminal-{candidate['scene_id']}".encode()
                ).hexdigest(),
                "runtime_seed": candidate["runtime_seed"],
                "material_ids": assignments_by_scene[row["scene_id"]],
                "selected_by_model_blind_prefix_rule": True,
            }
        )
    registry = {
        "scenes": realized_scenes,
        "attrition": attrition,
    }
    material_lock = {
        "materials": [
            {
                "id": row["material_id"],
                "source_asset_id": f"fresh_source_{index:02d}",
                "archive_sha256": hashlib.sha256(f"material-{index}".encode()).hexdigest(),
                "domains": [row["domain"]],
                "split": "confirmatory",
                "maps": {
                    name: {
                        "sha256": hashlib.sha256(
                            f"material-{index}-{name}".encode()
                        ).hexdigest(),
                        "bytes": 100 + index,
                    }
                    for name in ("basecolor", "normal", "roughness")
                },
            }
            for index, row in enumerate(design["materials"])
        ]
    }
    scene_index = {row["scene_id"]: int(row["scene_index"]) for row in design["scenes"]}
    operator_index = {operator: index for index, operator in enumerate(design["operators"])}
    scale_rows = []
    for context in design["scene_material_assignments"]:
        s_index = scene_index[context["scene_id"]]
        material_slot = int(context["material_slot"])
        for operator in design["operators"]:
            o_index = operator_index[operator]
            for replicate in range(16):
                offset = ((s_index * 2 + material_slot) * 11 + o_index) * 16 + replicate
                seed = 2_030_000_000 + offset
                group_id = f"scale_pair_{offset:05d}"
                severity = "moderate" if replicate < 8 else "severe"
                parameters = {
                    "lambda": (
                        0.15 + (replicate + 0.5) * 0.3 / 8
                        if replicate < 8
                        else 0.65 + (replicate - 8 + 0.5) * 0.3 / 8
                    ),
                    "operator": operator,
                }
                for condition in (
                    "anomaly",
                    "nominal_counterfactual",
                ):
                    nuisance = {
                        "profile_index": replicate,
                        "start_progress_m": 0.01,
                        "start_lateral_offset_m": 0.0,
                        "start_heading_offset_rad": 0.0,
                        "forward_speed_mps": 0.24,
                        "controller_target_lateral_offset_m": 0.0,
                        "physics_seed": seed,
                        "pair_shared": True,
                    }
                    scale_rows.append(
                        {
                            "counterfactual_group_id": group_id,
                            "condition": condition,
                            "scene_index": s_index,
                            "scene_id": context["scene_id"],
                            "scene_family": context["scene_id"],
                            "material_slot": material_slot,
                            "cluster_material": context["material_id"],
                            "material_asset_id": context["material_id"],
                            "target_operator": operator,
                            "operator_index": o_index,
                            "replicate_index": replicate,
                            "operator_seed": seed,
                            "severity_id": severity,
                            "parameter_interpolation": {"lambda": parameters["lambda"]},
                            "physical_nuisance": nuisance,
                            "physics_parameters": parameters,
                        }
                    )
    conflict_rows = []
    for cell_index, cell in enumerate(("T2_vision_decisive", "T3_proprio_decisive")):
        for context in design["scene_material_assignments"]:
            context_index = int(context["context_index"])
            s_index = scene_index[context["scene_id"]]
            for local_case in range(25):
                offset = cell_index * 1_500 + context_index * 25 + local_case
                case_id = f"{cell}_case_{offset:04d}"
                causes = (
                    ("O2_compliance", "O4_tether")
                    if cell_index == 0
                    else ("O7_visual_remap", "O8_invisible_collider")
                )
                nuisance = {
                    "profile_index": local_case % 16,
                    "start_progress_m": 0.01,
                    "start_lateral_offset_m": 0.0,
                    "start_heading_offset_rad": 0.0,
                    "forward_speed_mps": 0.24,
                    "controller_target_lateral_offset_m": 0.0,
                    "physics_seed": 2_040_000_000 + offset,
                    "pair_shared": True,
                }
                for cause in causes:
                    for view in range(3):
                        conflict_rows.append(
                            {
                                "cell": cell,
                                "case_id": case_id,
                                "case_seed": 2_040_000_000 + offset,
                                "scene_index": s_index,
                                "scene_id": context["scene_id"],
                                "scene_cluster": context["scene_id"],
                                "material_slot": int(context["material_slot"]),
                                "cluster_material": context["material_id"],
                                "material_family": context["material_id"],
                                "local_case_index": local_case,
                                "physical_nuisance": nuisance,
                                "cause_id": cause,
                                "view_id": f"view_{view}",
                            }
                        )
    return f0, registry, material_lock, scale_rows, conflict_rows


def test_full_freshness_contract_accepts_exact_f0_realization() -> None:
    f0, registry, materials, scale, conflict = _realized_fixture()
    checks, detail = audit_freshness(
        f0=f0,
        scene_registry=registry,
        material_lock=materials,
        scale_rows=scale,
        conflict_rows=conflict,
        prior=_empty_prior(),
    )
    assert all(checks.values()), {key: value for key, value in checks.items() if not value}
    assert detail["scale_pairs"] == 10_560
    assert detail["conflict_cases_by_cell"] == {
        "T2_vision_decisive": 1_500,
        "T3_proprio_decisive": 1_500,
    }


def test_freshness_contract_detects_prior_scene_collision() -> None:
    f0, registry, materials, scale, conflict = _realized_fixture()
    prior = _empty_prior()
    prior["scenes"].add(registry["scenes"][0]["scene_id"])
    checks, detail = audit_freshness(
        f0=f0,
        scene_registry=registry,
        material_lock=materials,
        scale_rows=scale,
        conflict_rows=conflict,
        prior=prior,
    )
    assert not checks["scene_ids_absent_from_prior_exposure"]
    assert detail["collisions"]["scenes"] == [registry["scenes"][0]["scene_id"]]


def test_prior_exposure_collector_reads_nested_hashes_and_vectors(
    tmp_path,
) -> None:
    path = tmp_path / "prior.jsonl"
    path.write_text(
        '{"scene_family":"old_scene","material_asset_id":"old_mat",'
        '"operator_seed":77,"physics_parameters":{"mu":0.1},'
        '"artifact_sha256":"' + "a" * 64 + '"}\n',
        encoding="utf-8",
    )
    exposure = _collect_exposure([path])
    assert exposure["scenes"] == {"old_scene"}
    assert exposure["materials"] == {"old_mat"}
    assert exposure["seeds"] == {77}
    assert exposure["content_hashes"] == {"a" * 64}
    assert len(exposure["operator_vectors"]) == 1
