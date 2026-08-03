#!/usr/bin/env python3
"""Fail-closed freshness audit for the unified-MoE confirmation corpus.

The audit proves that the realized scene registry, PBR lock, schedules, seeds,
and operator parameter vectors satisfy the pre-data F0 design and do not
intersect the registered development exposures.  It performs no model
inference and never reads attribution outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR_EXPOSURES = (
    "outputs/eval/realistic_a0_a7_v6/snapshots/snapshot_records.jsonl",
    "outputs/eval/c2_bidirectional_v5/development_features_six_scene/records.jsonl",
    "outputs/eval/c2_bidirectional_v5/formal_features/records.jsonl",
    "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl",
)
SCENE_KEYS = {
    "scene",
    "scene_id",
    "scene_family",
    "scene_cluster",
    "source_scene_id",
    "candidate_id",
}
MATERIAL_ID_KEYS = {
    "material",
    "material_id",
    "material_asset_id",
    "material_family",
    "source_asset_id",
}
SEED_KEYS = {
    "seed",
    "scene_seed",
    "source_scene_seed",
    "operator_seed",
    "physical_seed",
    "appearance_seed",
    "case_seed",
    "formal_episode_seed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        raise TypeError(path)
    for key in ("records", "rows", "scenes", "materials"):
        if isinstance(value.get(key), list):
            return [row for row in value[key] if isinstance(row, dict)]
    return [value]


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _collect_exposure(paths: list[Path]) -> dict[str, set[Any]]:
    exposure: dict[str, set[Any]] = {
        "scenes": set(),
        "materials": set(),
        "seeds": set(),
        "content_hashes": set(),
        "operator_vectors": set(),
    }
    for path in paths:
        for record in _records(path):
            parameters = record.get("physics_parameters")
            if isinstance(parameters, dict):
                exposure["operator_vectors"].add(_canonical_hash(parameters))
            for key, value in _walk(record):
                if key in SCENE_KEYS and isinstance(value, (str, int)):
                    exposure["scenes"].add(str(value))
                if key in MATERIAL_ID_KEYS and isinstance(value, (str, int)):
                    exposure["materials"].add(str(value))
                if key in SEED_KEYS and isinstance(value, int):
                    exposure["seeds"].add(int(value))
                if "sha256" in key.lower() and isinstance(value, str) and len(value) == 64:
                    exposure["content_hashes"].add(value)
    return exposure


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_frozen_path(display_path: str) -> Path:
    path = Path(display_path)
    return path if path.is_absolute() else ROOT / path


def validate_f0(path: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    manifest_path = path.resolve()
    sidecar = manifest_path.with_name("freeze_manifest.sha256")
    manifest = _json(manifest_path)
    expected_manifest_hash = sidecar.read_text(encoding="utf-8").split()[0]
    checks = {
        "manifest_sidecar_matches": (sha256_file(manifest_path) == expected_manifest_hash),
        "status_is_pre_generation_freeze": (
            manifest.get("status") == "frozen_before_new_scene_generation"
        ),
        "confirmatory_true": manifest.get("confirmatory") is True,
        "new_data_unavailable_at_freeze": (manifest.get("new_data_available_at_freeze") is False),
        "predictions_unavailable_at_freeze": (
            manifest.get("model_predictions_available_at_freeze") is False
        ),
        "fit_forbidden": (manifest.get("architecture", {}).get("fit_or_refit_permitted") is False),
        "threshold_strictly_greater": (
            manifest.get("architecture", {}).get("route_score_threshold") == 0.8
            and manifest.get("architecture", {}).get("threshold_comparison")
            == "strictly_greater_than"
        ),
        "vision_tie_break": (
            manifest.get("architecture", {}).get("equal_excess_tie_break") == "vision"
        ),
        "scale_count": (
            manifest.get("design", {}).get("scale", {}).get("counterfactual_pairs") == 10_560
        ),
        "conflict_counts": (
            manifest.get("design", {}).get("conflict", {}).get("cases_per_cell") == 1_500
        ),
    }
    frozen_files_valid = True
    for item in manifest.get("frozen_files", []):
        frozen_path = _resolve_frozen_path(str(item["path"]))
        frozen_files_valid = (
            frozen_files_valid
            and frozen_path.is_file()
            and frozen_path.stat().st_size == int(item["bytes"])
            and sha256_file(frozen_path) == str(item["sha256"])
        )
    checks["all_frozen_files_unchanged"] = frozen_files_valid
    return manifest, checks


def _material_content_hashes(materials: list[dict[str, Any]]) -> set[str]:
    values = set()
    for material in materials:
        for key, value in _walk(material):
            if "sha256" in key.lower() and isinstance(value, str) and len(value) == 64:
                values.add(value)
    return values


def _schedule_seed(row: dict[str, Any], candidates: tuple[str, ...]) -> int | None:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, int):
            return int(value)
    return None


def _candidate_seed(row: dict[str, Any]) -> int:
    value = row.get("source_seed", row.get("metric_geometry_seed"))
    if not isinstance(value, int):
        raise TypeError(f"candidate has no integer source/geometry seed: {row}")
    return int(value)


def _scene_admission_contract(
    f0: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    design = f0["design"]
    registered = [dict(row) for row in registry.get("scenes", [])]
    by_domain = {
        domain: sorted(
            (row for row in registered if str(row.get("domain")) == domain),
            key=lambda row: str(row.get("scene_id")),
        )
        for domain in ("life", "production", "wild")
    }
    selected_expected: dict[str, list[dict[str, Any]]] = {}
    prefix_valid = True
    attempt_binding_valid = True
    terminal_hashes_present = True
    for domain in ("life", "production", "wild"):
        stream = [dict(row) for row in design["scene_candidate_streams"][domain]]
        attempts = list(
            registry.get("attrition", {}).get(domain, {}).get("attempts", [])
        )
        if domain == "wild":
            expected_attempt_count = 10
        else:
            admitted_count = 0
            expected_attempt_count = 0
            for candidate in stream:
                expected_attempt_count += 1
                if (
                    expected_attempt_count <= len(attempts)
                    and attempts[expected_attempt_count - 1].get("admitted") is True
                ):
                    admitted_count += 1
                if admitted_count == 10:
                    break
        prefix_valid = prefix_valid and len(attempts) == expected_attempt_count
        admitted_candidates: list[dict[str, Any]] = []
        for index, attempt in enumerate(attempts):
            if index >= len(stream):
                attempt_binding_valid = False
                continue
            candidate = attempt.get("candidate")
            attempt_binding_valid = (
                attempt_binding_valid
                and attempt.get("candidate_index") == index
                and isinstance(candidate, dict)
                and candidate == stream[index]
            )
            terminal_hash = attempt.get("terminal_audit_sha256")
            terminal_hashes_present = (
                terminal_hashes_present
                and isinstance(terminal_hash, str)
                and len(terminal_hash) == 64
            )
            if attempt.get("admitted") is True:
                admitted_candidates.append(stream[index])
        selected_expected[domain] = admitted_candidates[:10]
        if (
            len(selected_expected[domain]) != 10
            or (domain == "wild" and len(admitted_candidates) != 10)
        ):
            prefix_valid = False

    slot_binding_valid = True
    seen_candidates: set[str] = set()
    geometry_hashes: set[str] = set()
    for domain in ("life", "production", "wild"):
        rows = by_domain[domain]
        expected = selected_expected.get(domain, [])
        if len(rows) != 10 or len(expected) != 10:
            slot_binding_valid = False
            continue
        for slot, (row, candidate) in enumerate(zip(rows, expected, strict=True)):
            expected_alias = f"confirm_v1_{domain}_scene_{slot:02d}"
            candidate_id = str(candidate["scene_id"])
            actual_seed = row.get(
                "metric_geometry_seed" if domain == "wild" else "source_scene_seed"
            )
            geometry_hash = row.get("geometry_hash")
            slot_binding_valid = (
                slot_binding_valid
                and row.get("scene_id") == expected_alias
                and row.get("candidate_id") == candidate_id
                and row.get("source_scene_id") == candidate_id
                and actual_seed == _candidate_seed(candidate)
                and row.get("runtime_seed") == candidate.get("runtime_seed")
                and row.get("selected_by_model_blind_prefix_rule") is True
                and isinstance(geometry_hash, str)
                and len(geometry_hash) == 64
                and isinstance(row.get("terminal_scene_admission_sha256"), str)
                and len(str(row.get("terminal_scene_admission_sha256"))) == 64
            )
            seen_candidates.add(candidate_id)
            if isinstance(geometry_hash, str):
                geometry_hashes.add(geometry_hash)

    checks = {
        "admission_attempts_are_gap_free_frozen_prefix": prefix_valid,
        "admission_attempts_bind_exact_frozen_candidates": attempt_binding_valid,
        "admission_terminal_hashes_present": terminal_hashes_present,
        "scene_alias_candidate_seed_runtime_domain_binding_exact": slot_binding_valid,
        "selected_source_scene_ids_unique": len(seen_candidates) == 30,
        "selected_geometry_hashes_unique": len(geometry_hashes) == 30,
    }
    return checks, {
        "attempted_by_domain": {
            domain: len(
                registry.get("attrition", {}).get(domain, {}).get("attempts", [])
            )
            for domain in ("life", "production", "wild")
        },
        "selected_source_scene_ids": sorted(seen_candidates),
        "unique_geometry_hashes": len(geometry_hashes),
    }


def _material_contract(
    design: dict[str, Any],
    registry: dict[str, Any],
    materials: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    planned = {str(row["material_id"]): str(row["domain"]) for row in design["materials"]}
    indexed = {str(row.get("id")): row for row in materials}
    source_ids = [str(row.get("source_asset_id", "")) for row in materials]
    required_maps = ("basecolor", "normal", "roughness")
    evidence_valid = True
    domains_valid = True
    for material_id, expected_domain in planned.items():
        row = indexed.get(material_id, {})
        domains_valid = (
            domains_valid
            and row.get("domains") == [expected_domain]
            and row.get("split") == "confirmatory"
        )
        archive_hash = row.get("archive_sha256")
        maps = row.get("maps", {})
        evidence_valid = (
            evidence_valid
            and isinstance(archive_hash, str)
            and len(archive_hash) == 64
            and all(
                isinstance(maps.get(name), dict)
                and isinstance(maps[name].get("sha256"), str)
                and len(maps[name]["sha256"]) == 64
                and int(maps[name].get("bytes", 0)) > 0
                for name in required_maps
            )
        )
    expected_assignments = {
        str(row["scene_id"]): [
            str(item["material_id"])
            for item in sorted(
                (
                    candidate
                    for candidate in design["scene_material_assignments"]
                    if candidate["scene_id"] == row["scene_id"]
                ),
                key=lambda candidate: int(candidate["material_slot"]),
            )
        ]
        for row in design["scenes"]
    }
    actual_assignments = {
        str(row.get("scene_id")): [str(value) for value in row.get("material_ids", [])]
        for row in registry.get("scenes", [])
    }
    checks = {
        "material_alias_domain_binding_exact": (
            set(indexed) == set(planned) and domains_valid
        ),
        "material_source_asset_ids_unique_and_present": (
            len(source_ids) == 30
            and "" not in source_ids
            and len(set(source_ids)) == 30
        ),
        "each_material_has_archive_and_required_map_hashes": evidence_valid,
        "scene_material_ring_assignment_matches_f0": (
            actual_assignments == expected_assignments
        ),
    }
    return checks, {
        "source_asset_ids": sorted(source_ids),
        "scene_material_assignments": actual_assignments,
    }


def _row_scene(row: dict[str, Any]) -> str:
    return str(
        row.get("scene_id")
        or row.get("scene_cluster")
        or row.get("scene_family")
        or ""
    )


def _row_material(row: dict[str, Any]) -> str:
    return str(
        row.get("cluster_material")
        or row.get("material_context_id")
        or row.get("material_id")
        or row.get("material_asset_id")
        or row.get("material_family")
        or ""
    )


def _scale_contract(
    design: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any], set[int], set[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("counterfactual_group_id", ""))].append(row)
    scene_index = {str(row["scene_id"]): int(row["scene_index"]) for row in design["scenes"]}
    operator_index = {operator: index for index, operator in enumerate(design["operators"])}
    assignments = {
        (str(row["scene_id"]), int(row["material_slot"])): str(row["material_id"])
        for row in design["scene_material_assignments"]
    }
    expected = {
        (
            scene_id,
            material_id,
            material_slot,
            operator,
            op_index,
            replicate,
            2_030_000_000
            + (((s_index * 2 + material_slot) * 11 + op_index) * 16 + replicate),
        )
        for scene_id, s_index in scene_index.items()
        for material_slot in (0, 1)
        for material_id in (assignments[(scene_id, material_slot)],)
        for operator, op_index in operator_index.items()
        for replicate in range(16)
    }
    actual: set[tuple[str, str, int, str, int, int, int]] = set()
    exact_pair_structure = "" not in groups
    nuisance_contract = True
    seed_formula = True
    parameter_hashes: set[str] = set()
    seeds: set[int] = set()
    exact_nuisance_keys = {
        "profile_index",
        "start_progress_m",
        "start_lateral_offset_m",
        "start_heading_offset_rad",
        "forward_speed_mps",
        "controller_target_lateral_offset_m",
        "physics_seed",
        "pair_shared",
    }
    for group_rows in groups.values():
        first = group_rows[0]
        scene_id = _row_scene(first)
        material_id = _row_material(first)
        material_slot = first.get("material_slot")
        operator = str(first.get("target_operator", ""))
        op_index = first.get("operator_index")
        replicate = first.get("replicate_index")
        seed = _schedule_seed(first, ("operator_seed", "physical_seed"))
        if all(
            isinstance(value, int)
            for value in (material_slot, op_index, replicate, seed)
        ):
            actual.add(
                (
                    scene_id,
                    material_id,
                    int(material_slot),
                    operator,
                    int(op_index),
                    int(replicate),
                    int(seed),
                )
            )
            seeds.add(int(seed))
            expected_seed = 2_030_000_000 + (
                (
                    (scene_index.get(scene_id, -1) * 2 + int(material_slot)) * 11
                    + operator_index.get(operator, -1)
                )
                * 16
                + int(replicate)
            )
            seed_formula = seed_formula and int(seed) == expected_seed
        else:
            seed_formula = False
        nuisance = first.get("physical_nuisance")
        nuisance_contract = (
            nuisance_contract
            and isinstance(nuisance, dict)
            and set(nuisance) == exact_nuisance_keys
            and nuisance.get("physics_seed") == seed
            and nuisance.get("pair_shared") is True
        )
        shared_keys = (
            "scene_index",
            "scene_id",
            "scene_family",
            "material_slot",
            "cluster_material",
            "target_operator",
            "operator_index",
            "replicate_index",
            "operator_seed",
            "severity_id",
            "parameter_interpolation",
            "physical_nuisance",
        )
        exact_pair_structure = (
            exact_pair_structure
            and len(group_rows) == 2
            and {str(row.get("condition")) for row in group_rows}
            == {"anomaly", "nominal_counterfactual"}
            and all(
                all(row.get(key) == first.get(key) for key in shared_keys)
                for row in group_rows
            )
        )
        for row in group_rows:
            if (
                row.get("condition") == "anomaly"
                and isinstance(row.get("physics_parameters"), dict)
            ):
                parameter_hashes.add(_canonical_hash(row["physics_parameters"]))
    checks = {
        "scale_expected_tuple_ledger_exact": actual == expected,
        "scale_seed_formula_bound_per_tuple": seed_formula,
        "scale_pairs_have_exact_shared_counterfactual_structure": exact_pair_structure,
        "scale_physical_nuisance_contract_exact": nuisance_contract,
        "scale_each_scene_material_operator_has_16_pairs": (
            len(actual) == 60 * 11 * 16
            and len(Counter((row[0], row[1], row[3]) for row in actual)) == 60 * 11
            and all(
                value == 16
                for value in Counter(
                    (row[0], row[1], row[3]) for row in actual
                ).values()
            )
        ),
    }
    detail = {
        "pairs": len(groups),
        "expected_tuples": len(expected),
        "actual_tuples": len(actual),
        "missing_tuples": len(expected - actual),
        "unexpected_tuples": len(actual - expected),
    }
    return checks, detail, seeds, parameter_hashes


def _conflict_contract(
    design: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any], set[int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("case_id", ""))].append(row)
    scene_index = {str(row["scene_id"]): int(row["scene_index"]) for row in design["scenes"]}
    assignments = {
        (str(row["scene_id"]), int(row["material_slot"])): (
            int(row["context_index"]),
            str(row["material_id"]),
        )
        for row in design["scene_material_assignments"]
    }
    cells = ("T2_vision_decisive", "T3_proprio_decisive")
    expected = {
        (
            cell,
            scene_id,
            material_id,
            material_slot,
            local_case,
            2_040_000_000 + cell_index * 1_500 + context_index * 25 + local_case,
        )
        for cell_index, cell in enumerate(cells)
        for scene_id in scene_index
        for material_slot in (0, 1)
        for context_index, material_id in (assignments[(scene_id, material_slot)],)
        for local_case in range(25)
    }
    actual: set[tuple[str, str, str, int, int, int]] = set()
    exact_structure = "" not in groups
    nuisance_contract = True
    seed_formula = True
    case_ids_unique_across_cells = True
    seeds: set[int] = set()
    case_to_cell: dict[str, str] = {}
    for case_id, case_rows in groups.items():
        first = case_rows[0]
        cell = str(first.get("cell", ""))
        scene_id = _row_scene(first)
        material_id = _row_material(first)
        material_slot = first.get("material_slot")
        local_case = first.get("local_case_index")
        seed = _schedule_seed(first, ("case_seed", "physical_seed", "operator_seed"))
        previous = case_to_cell.setdefault(case_id, cell)
        case_ids_unique_across_cells = case_ids_unique_across_cells and previous == cell
        if all(isinstance(value, int) for value in (material_slot, local_case, seed)):
            actual.add(
                (
                    cell,
                    scene_id,
                    material_id,
                    int(material_slot),
                    int(local_case),
                    int(seed),
                )
            )
            seeds.add(int(seed))
            context = assignments.get((scene_id, int(material_slot)), (-1, ""))[0]
            expected_seed = (
                2_040_000_000
                + (cells.index(cell) if cell in cells else -1) * 1_500
                + context * 25
                + int(local_case)
            )
            seed_formula = seed_formula and int(seed) == expected_seed
        else:
            seed_formula = False
        cause_view = {
            (
                str(row.get("cause_id") or row.get("target_operator") or ""),
                str(row.get("view_id") or row.get("appearance_view_id") or ""),
            )
            for row in case_rows
        }
        causes = {cause for cause, _view in cause_view}
        views = {view for _cause, view in cause_view}
        expected_causes = (
            {"O2_compliance", "O4_tether"}
            if cell == "T2_vision_decisive"
            else {"O7_visual_remap", "O8_invisible_collider"}
        )
        nuisance = first.get("physical_nuisance")
        nuisance_contract = (
            nuisance_contract
            and isinstance(nuisance, dict)
            and set(nuisance)
            == {
                "profile_index",
                "start_progress_m",
                "start_lateral_offset_m",
                "start_heading_offset_rad",
                "forward_speed_mps",
                "controller_target_lateral_offset_m",
                "physics_seed",
                "pair_shared",
            }
            and nuisance.get("physics_seed") == seed
            and nuisance.get("pair_shared") is True
        )
        shared_keys = (
            "cell",
            "case_id",
            "case_seed",
            "scene_index",
            "scene_id",
            "scene_cluster",
            "material_slot",
            "cluster_material",
            "local_case_index",
            "physical_nuisance",
        )
        exact_structure = (
            exact_structure
            and len(case_rows) == 6
            and causes == expected_causes
            and len(views) == 3
            and len(cause_view) == 6
            and all(
                all(row.get(key) == first.get(key) for key in shared_keys)
                for row in case_rows
            )
        )
    per_context = Counter((r[0], r[1], r[2]) for r in actual)
    checks = {
        "conflict_expected_tuple_ledger_exact": actual == expected,
        "conflict_seed_formula_bound_per_tuple": seed_formula,
        "conflict_case_ids_globally_unique": (
            case_ids_unique_across_cells and len(groups) == len(case_to_cell)
        ),
        "conflict_each_cell_context_has_25_cases": (
            len(per_context) == 2 * 60
            and all(value == 25 for value in per_context.values())
        ),
        "conflict_six_record_cause_view_cartesian_structure": exact_structure,
        "conflict_physical_nuisance_contract_exact": nuisance_contract,
    }
    detail = {
        "cases": len(groups),
        "expected_tuples": len(expected),
        "actual_tuples": len(actual),
        "missing_tuples": len(expected - actual),
        "unexpected_tuples": len(actual - expected),
        "cases_by_cell": dict(Counter(row[0] for row in actual)),
    }
    return checks, detail, seeds


def audit_freshness(
    *,
    f0: dict[str, Any],
    scene_registry: dict[str, Any],
    material_lock: dict[str, Any],
    scale_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    prior: dict[str, set[Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    design = f0["design"]
    planned_scenes = {str(row["scene_id"]): row for row in design["scenes"]}
    planned_materials = {str(row["material_id"]): row for row in design["materials"]}
    scenes = [dict(row) for row in scene_registry.get("scenes", [])]
    materials = [dict(row) for row in material_lock.get("materials", [])]
    actual_scene_ids = {str(row.get("scene_id") or row.get("scene_family")) for row in scenes}
    actual_material_ids = {str(row.get("material_id") or row.get("id")) for row in materials}
    actual_material_source_ids = {
        str(row["source_asset_id"])
        for row in materials
        if row.get("source_asset_id") is not None
    }
    actual_scene_seeds = {
        int(
            row.get("source_scene_seed")
            or row.get("metric_geometry_seed")
            or row.get("scene_seed")
        )
        for row in scenes
        if isinstance(
            row.get("source_scene_seed")
            or row.get("metric_geometry_seed")
            or row.get("scene_seed"),
            int,
        )
    }
    actual_source_scene_ids = {
        str(
            row.get("source_scene_id")
            or row.get("candidate_id")
            or row.get("scene_id")
        )
        for row in scenes
    }
    new_material_hashes = _material_content_hashes(materials)
    scene_checks, scene_detail = _scene_admission_contract(f0, scene_registry)
    material_checks, material_detail = _material_contract(
        design, scene_registry, materials
    )
    scale_checks, scale_detail, scale_pair_seeds, scale_parameter_hashes = (
        _scale_contract(design, scale_rows)
    )
    conflict_checks, conflict_detail, conflict_seeds = _conflict_contract(
        design, conflict_rows
    )
    scale_scene_ids = {_row_scene(row) for row in scale_rows}
    scale_material_ids = {_row_material(row) for row in scale_rows}
    conflict_scene_ids = {_row_scene(row) for row in conflict_rows}
    conflict_material_ids = {_row_material(row) for row in conflict_rows}
    checks = {
        "exactly_30_registered_scenes": len(scenes) == 30,
        "registered_scene_ids_match_f0": (actual_scene_ids == set(planned_scenes)),
        **scene_checks,
        "scene_ids_absent_from_prior_exposure": (not (actual_scene_ids & prior["scenes"])),
        "source_scene_ids_absent_from_prior_exposure": (
            not (actual_source_scene_ids & prior["scenes"])
        ),
        "scene_seeds_absent_from_prior_exposure": (not (actual_scene_seeds & prior["seeds"])),
        "exactly_30_registered_materials": len(materials) == 30,
        "registered_material_ids_match_f0": (actual_material_ids == set(planned_materials)),
        **material_checks,
        "material_ids_absent_from_prior_exposure": (not (actual_material_ids & prior["materials"])),
        "material_source_ids_present": (
            len(actual_material_source_ids) == 30
            and len(materials) == len(actual_material_source_ids)
        ),
        "material_source_ids_absent_from_prior_exposure": (
            not (actual_material_source_ids & prior["materials"])
        ),
        "material_content_hashes_present": (len(new_material_hashes) >= 120),
        "material_content_hashes_absent_from_prior_exposure": (
            not (new_material_hashes & prior["content_hashes"])
        ),
        **scale_checks,
        "scale_uses_exact_f0_scenes": (scale_scene_ids == set(planned_scenes)),
        "scale_uses_exact_f0_materials": (scale_material_ids == set(planned_materials)),
        "scale_uses_all_11_operators": (
            {str(row.get("target_operator")) for row in scale_rows} == set(design["operators"])
        ),
        "scale_seeds_absent_from_prior_exposure": (not (scale_pair_seeds & prior["seeds"])),
        "operator_vectors_absent_from_prior_exposure": (
            bool(scale_parameter_hashes)
            and not (scale_parameter_hashes & prior["operator_vectors"])
        ),
        **conflict_checks,
        "conflict_uses_exact_f0_scenes": (conflict_scene_ids == set(planned_scenes)),
        "conflict_uses_exact_f0_materials": (conflict_material_ids == set(planned_materials)),
        "conflict_seeds_absent_from_prior_exposure": (not (conflict_seeds & prior["seeds"])),
    }
    detail = {
        "registered_scenes": len(scenes),
        "registered_source_scenes": len(actual_source_scene_ids),
        "registered_materials": len(materials),
        "registered_material_source_assets": len(actual_material_source_ids),
        "new_material_content_hashes": len(new_material_hashes),
        "scene_admission": scene_detail,
        "material_contract": material_detail,
        "scale_contract": scale_detail,
        "conflict_contract": conflict_detail,
        "scale_pairs": scale_detail["pairs"],
        "scale_pair_seeds": len(scale_pair_seeds),
        "scale_operator_parameter_vectors": len(scale_parameter_hashes),
        "conflict_cases_by_cell": dict(
            sorted(conflict_detail["cases_by_cell"].items())
        ),
        "conflict_case_seeds": len(conflict_seeds),
        "collisions": {
            "scenes": sorted(actual_scene_ids & prior["scenes"]),
            "source_scenes": sorted(actual_source_scene_ids & prior["scenes"]),
            "materials": sorted(actual_material_ids & prior["materials"]),
            "material_source_assets": sorted(
                actual_material_source_ids & prior["materials"]
            ),
            "seeds": sorted(
                (actual_scene_seeds | scale_pair_seeds | conflict_seeds) & prior["seeds"]
            ),
            "material_content_hashes": sorted(new_material_hashes & prior["content_hashes"]),
            "operator_vectors": sorted(scale_parameter_hashes & prior["operator_vectors"]),
        },
    }
    return checks, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--scale-schedule", type=Path, required=True)
    parser.add_argument("--conflict-schedule", type=Path, required=True)
    parser.add_argument(
        "--prior-exposure",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite freshness audit: {out}")
    f0, f0_checks = validate_f0(args.f0_manifest)
    prior_paths = [(ROOT / value).resolve() for value in DEFAULT_PRIOR_EXPOSURES]
    prior_paths.extend(path.resolve() for path in args.prior_exposure)
    prior = _collect_exposure(prior_paths)
    freshness_checks, detail = audit_freshness(
        f0=f0,
        scene_registry=_json(args.scene_registry),
        material_lock=_json(args.material_lock),
        scale_rows=_records(args.scale_schedule),
        conflict_rows=_records(args.conflict_schedule),
        prior=prior,
    )
    checks = {
        **{f"f0/{key}": value for key, value in f0_checks.items()},
        **{f"freshness/{key}": value for key, value in freshness_checks.items()},
    }
    payload = {
        "schema_version": "kinofail.confirmatory-freshness-audit.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "detail": detail,
        "source_sha256": {
            "f0_manifest": sha256_file(args.f0_manifest),
            "scene_registry": sha256_file(args.scene_registry),
            "material_lock": sha256_file(args.material_lock),
            "scale_schedule": sha256_file(args.scale_schedule),
            "conflict_schedule": sha256_file(args.conflict_schedule),
            "prior_exposures": {str(path): sha256_file(path) for path in prior_paths},
            "auditor": sha256_file(Path(__file__).resolve()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
