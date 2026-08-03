#!/usr/bin/env python3
"""Seal the independent reconfirmation design before any new asset exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = (
    ROOT / "configs/data/kinofail_unified_reconfirmation_design_v2.json"
)
DEFAULT_OUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0"
ORIGINAL_F0_SHA256 = (
    "8eced1a4fa1ea7babbc6e6d5d3852a3a5c85896594dfd160b3279979820c80e5"
)
ORIGINAL_F1_SHA256 = (
    "5e53b941e37dfc8693fe23c17faba9f452d043b9fd8d050e0c9cd063e5f908e1"
)
ORIGINAL_F2_SHA256 = (
    "a2922e4e9859c04c64903c96f6f682f1e4654bab109286bcbe6cf1a3cac61968"
)
INVALID_RECEIPT_SHA256 = (
    "d51db740bb0d8ccb1751d7d06eab68ec8a99f3aa043e84c022354213414c3cd1"
)
QA_RECEIPT_SHA256 = (
    "99ff8d22a8108e2f3e3756b44947b67d8c64ffc3ac186a63f7ba532e9cfec079"
)
ARCHIVE_REGISTRY_SHA256 = (
    "52c237354621358e1e4ff29a6df925bd35eae949ec6636ced58a5b79eaf2fd44"
)
EXPECTED_CHECKPOINT_SHA256 = [
    "ef357e18cad5ea304a9e14fe053de9aa32628575b2566c3583ac4c572f852054",
    "957d146b097429eccb2fad202469ff2dc52843e41caf88a421a8054c95934b3e",
    "171298b5dd29f165ff39432c8313d915802d3aa672f72c756a785a693f78605a",
    "9c7cc2bd2dc2a1270a679180c3706f4046086a0fd367b565d004499ecc41dcd4",
    "4646d0683ed0958ceffff41868638dbef9743ad5db41720a9583136796fe40fd",
]
DOMAINS = ("life", "production", "wild")
OPERATORS = [
    "O1_mu_field",
    "O2_compliance",
    "O3_collapse",
    "O4_tether",
    "O5_payload",
    "O6_push",
    "O7_visual_remap",
    "O8_invisible_collider",
    "O9_high_centering",
    "O10_effort_decay",
    "O11_obs_bias",
]
FORBIDDEN_NEW_DATA_ROOTS = (
    "outputs/kinofail_reconfirmation_v2",
    "outputs/assets/terrain_pbr_confirmatory_v2",
    "outputs/eval/unified_moe_v3_reconfirmation_v2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _resolve_archived(value: str) -> Path:
    """Resolve provenance paths after the invalid pilot was atomically archived."""

    path = Path(value)
    try:
        relative = path.resolve().relative_to(
            (ROOT / "outputs/kinofail_confirmatory_v1").resolve()
        )
    except ValueError:
        return _resolve(value)
    return (
        ROOT
        / "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725"
        / relative
    ).resolve()


def _require_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"authority hash mismatch: {path}")
    return _json(path)


def _walk_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(_walk_key(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_key(child, key))
    return found


def _registered_files(design: dict[str, Any]) -> list[Path]:
    values = design.get("frozen_files")
    if not isinstance(values, list) or not values:
        raise RuntimeError("frozen_files must be a nonempty list")
    paths = [_resolve(str(value)) for value in values]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(paths), key=str)


def _archive_sources(
    archive: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [dict(row) for row in archive.get("scenes", [])]
    if len(rows) != 30:
        raise RuntimeError("archive registry must contain 30 scenes")
    return (
        {str(row["scene_id"]): row for row in rows},
        {str(row["source_scene_id"]): row for row in rows},
    )


def _validate_exposure_and_reuse(
    *,
    scene_config: dict[str, Any],
    exclusions: dict[str, Any],
    invalid_receipt: dict[str, Any],
    archive: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_slot, by_source = _archive_sources(archive)
    started_slots = set(
        str(value)
        for key in ("t2_scene_directories_started", "t3_scenes_started")
        for value in invalid_receipt["partial_exposure"][key]
    )
    exclusion_rows = [dict(row) for row in exclusions["records"]]
    exclusion_slots = {str(row["scene_id"]) for row in exclusion_rows}
    exclusion_sources = {str(row["source_scene_id"]) for row in exclusion_rows}
    mapped_sources = {str(by_slot[slot]["source_scene_id"]) for slot in started_slots}
    configured_sources = set(
        str(value)
        for value in scene_config["independence_contract"][
            "excluded_source_scene_ids"
        ]
    )
    configured_slots = set(
        str(value)
        for value in scene_config["independence_contract"][
            "excluded_formal_slot_ids"
        ]
    )
    if not (
        started_slots
        == exclusion_slots
        == configured_slots
        and mapped_sources
        == exclusion_sources
        == configured_sources
    ):
        raise RuntimeError("invalid-pilot exposure ledgers disagree")

    reused = [
        dict(row) for row in scene_config["unexposed_original_f0_scenes"]
    ]
    if len(reused) != 21:
        raise RuntimeError("exactly 21 original-F0 scenes must be unexposed")
    expected_unexposed_slots = set(by_slot) - started_slots
    if {str(row["original_slot_id"]) for row in reused} != expected_unexposed_slots:
        raise RuntimeError("reuse list is not the exact unexposed complement")
    for row in reused:
        archived = by_slot[str(row["original_slot_id"])]
        if (
            row["source_scene_id"] != archived["source_scene_id"]
            or row["domain"] != archived["domain"]
            or int(row["archive_registry_index"])
            != archive["scenes"].index(archived)
            or row["source_scene_id"] in exclusion_sources
        ):
            raise RuntimeError(f"invalid unexposed mapping: {row}")
        for path_key, hash_key in (
            ("episode_usd", "episode_sha256"),
            ("compiled_audit", "compiled_audit_sha256"),
            ("terminal_scene_admission", "terminal_scene_admission_sha256"),
        ):
            path = _resolve_archived(str(archived[path_key]))
            if not path.is_file() or _sha256(path) != str(archived[hash_key]):
                raise RuntimeError(f"archived scene evidence drift: {path}")

    invalid_root = (
        ROOT
        / "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725"
    )
    actual_runtime_slots = {
        path.name
        for path in (invalid_root / "c2_t2").glob("confirm_v1_*_scene_*")
        if any(path.rglob("manifest.json"))
    }
    actual_runtime_slots.update(
        path.name
        for path in (invalid_root / "corpus").glob("confirm_v1_*_scene_*")
        if any(path.glob("c2_t3/**/manifest.json"))
    )
    if not actual_runtime_slots <= started_slots:
        raise RuntimeError("unreported invalid-pilot runtime scene exposure")
    return reused, {
        "started_formal_slots": sorted(started_slots),
        "excluded_source_scene_ids": sorted(exclusion_sources),
        "unexposed_original_f0_slots": sorted(expected_unexposed_slots),
        "runtime_directory_slots": sorted(actual_runtime_slots),
    }


def _candidate_streams(
    *,
    reused: list[dict[str, Any]],
    scene_config: dict[str, Any],
    archive: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    _by_slot, by_source = _archive_sources(archive)
    streams: dict[str, list[dict[str, Any]]] = {}
    reused_order = {
        "life": [
            "confirm_v1_life_scene_08",
            "confirm_v1_life_scene_09",
        ],
        "production": [
            "confirm_v1_production_scene_00",
            "confirm_v1_production_scene_01",
            "confirm_v1_production_scene_02",
            "confirm_v1_production_scene_03",
            "confirm_v1_production_scene_04",
            "confirm_v1_production_scene_05",
            "confirm_v1_production_scene_07",
            "confirm_v1_production_scene_08",
            "confirm_v1_production_scene_09",
        ],
        "wild": [f"confirm_v1_wild_scene_{index:02d}" for index in range(10)],
    }
    reused_by_slot = {str(row["original_slot_id"]): row for row in reused}
    for domain in DOMAINS:
        stream: list[dict[str, Any]] = []
        for slot in reused_order[domain]:
            mapping = reused_by_slot[slot]
            archived = by_source[str(mapping["source_scene_id"])]
            candidate = {
                "scene_id": str(archived["source_scene_id"]),
                "domain": domain,
                "runtime_seed": int(archived["runtime_seed"]),
                "origin": "original_f0_unexposed",
                "original_slot_id": slot,
                "archive_registry_index": int(mapping["archive_registry_index"]),
            }
            if domain == "wild":
                candidate.update(
                    {
                        "metric_geometry_seed": int(
                            archived["metric_geometry_seed"]
                        ),
                        "hdri_id": str(archived["hdri_id"]),
                    }
                )
            else:
                candidate.update(
                    {
                        "source_seed": int(archived["source_scene_seed"]),
                        "room_type": str(archived["room_type"]),
                    }
                )
            stream.append(candidate)
        if domain != "wild":
            stream.extend(
                {
                    **dict(row),
                    "origin": "new_gap_free_candidate",
                }
                for row in scene_config["indoor_candidates"]
                if str(row["domain"]) == domain
            )
        streams[domain] = stream
    if not (
        len(streams["life"]) == 22
        and len(streams["production"]) == 19
        and len(streams["wild"]) == 10
    ):
        raise RuntimeError("reconfirmation candidate stream size drift")
    return streams


def _validate_fresh_seeds(
    scene_config: dict[str, Any], archive: dict[str, Any]
) -> dict[str, Any]:
    new_rows = [dict(row) for row in scene_config["indoor_candidates"]]
    source_seeds = [int(row["source_seed"]) for row in new_rows]
    runtime_seeds = [int(row["runtime_seed"]) for row in new_rows]
    old_seed_values = {
        int(value)
        for row in archive["scenes"]
        for value in (
            row.get("source_scene_seed"),
            row.get("metric_geometry_seed"),
            row.get("runtime_seed"),
        )
        if isinstance(value, int)
    }
    if (
        len(source_seeds) != len(set(source_seeds))
        or len(runtime_seeds) != len(set(runtime_seeds))
        or set(source_seeds) & set(runtime_seeds)
        or (set(source_seeds) | set(runtime_seeds)) & old_seed_values
    ):
        raise RuntimeError("new scene seeds are not unique and legacy-disjoint")
    new_source_ids = {str(row["scene_id"]) for row in new_rows}
    old_source_ids = {str(row["source_scene_id"]) for row in archive["scenes"]}
    if new_source_ids & old_source_ids:
        raise RuntimeError("new scene IDs intersect original F0 sources")
    return {
        "new_source_seed_count": len(source_seeds),
        "new_runtime_seed_count": len(runtime_seeds),
        "new_scene_id_count": len(new_source_ids),
        "old_scene_seed_count": len(old_seed_values),
    }


def _validate_materials(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    materials = [dict(row) for row in catalog["materials"]]
    ids = [str(row["id"]) for row in materials]
    sources = [str(row["source_asset_id"]) for row in materials]
    if (
        len(materials) != 30
        or len(set(ids)) != 30
        or len(set(sources)) != 30
        or any(
            ids.count(f"confirm_v2_{domain}_pbr_{index:02d}") != 1
            for domain in DOMAINS
            for index in range(10)
        )
    ):
        raise RuntimeError("v2 material catalog structure drift")

    prior_sources: set[str] = set()
    prior_paths = sorted(
        item
        for item in (
            list((ROOT / "configs/data").glob("*terrain_pbr*.yaml"))
            + list((ROOT / "outputs/assets").glob("*/terrain_assets.lock.json"))
        )
        if item.resolve() != path.resolve()
    )
    for prior_path in prior_paths:
        try:
            value = (
                yaml.safe_load(prior_path.read_text(encoding="utf-8"))
                if prior_path.suffix in {".yaml", ".yml"}
                else _json(prior_path)
            )
        except (OSError, ValueError, yaml.YAMLError):
            continue
        prior_sources.update(
            str(value)
            for value in _walk_key(value, "source_asset_id")
            if isinstance(value, (str, int))
        )
    collisions = set(sources) & prior_sources
    if collisions:
        raise RuntimeError(f"v2 material sources are not fresh: {sorted(collisions)}")
    return materials, {
        "material_count": 30,
        "source_asset_count": 30,
        "prior_catalog_count": len(prior_paths),
        "prior_source_asset_count": len(prior_sources),
        "source_asset_collisions": [],
    }


def _audit_checkpoints(original_f0: dict[str, Any]) -> list[dict[str, Any]]:
    audits = deepcopy(original_f0["architecture"]["checkpoint_audits"])
    hashes = [str(row["sha256"]) for row in audits]
    if hashes != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("original F0 checkpoint order/hash drift")
    for row in audits:
        path = _resolve(str(row["path"]))
        if _sha256(path) != str(row["sha256"]):
            raise RuntimeError(f"checkpoint changed: {path}")
        with path.open("rb") as stream:
            model = pickle.load(stream)  # noqa: S301 - local hash-pinned artifact
        if (
            getattr(model, "route_thresholds", None)
            != {"vision": 0.8, "joint": 0.8}
            or list(getattr(model, "classes", [])) != row["classes"]
        ):
            raise RuntimeError(f"checkpoint contract mismatch: {path}")
    return audits


def _internal_design(
    *,
    external: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(DOMAINS):
        for local in range(10):
            index = domain_index * 10 + local
            scene_id = f"confirm_v2_{domain}_scene_{local:02d}"
            material_id = f"confirm_v2_{domain}_pbr_{local:02d}"
            scenes.append(
                {
                    "scene_index": index,
                    "scene_id": scene_id,
                    "domain": domain,
                    "admission_slot": local,
                    "source_selection": (
                        "frozen_unexposed_then_gap_free_new_prefix"
                    ),
                }
            )
            materials.append(
                {
                    "material_index": index,
                    "material_id": material_id,
                    "domain": domain,
                    "must_be_new_source_asset": True,
                }
            )
            for material_slot, local_material in enumerate(
                (local, (local + 5) % 10)
            ):
                assignments.append(
                    {
                        "context_index": len(assignments),
                        "scene_id": scene_id,
                        "material_id": (
                            f"confirm_v2_{domain}_pbr_{local_material:02d}"
                        ),
                        "material_slot": material_slot,
                        "domain": domain,
                    }
                )
    scale = deepcopy(external["scale"])
    scale["fresh_seeds_per_scene_material_operator"] = 16
    scale["seed_rule"] = {
        "formula": external["scale"]["seed_formula"],
        "replicate_indices": list(range(16)),
        "moderate_replicate_indices": list(range(8)),
        "severe_replicate_indices": list(range(8, 16)),
        "moderate_lambda_values": external["scale"][
            "moderate_lambda_points"
        ],
        "severe_lambda_values": external["scale"]["hard_lambda_points"],
        "operator_interpolation": (
            "p(lambda)=(1-lambda)*moderate+lambda*severe"
        ),
        "old_endpoint_lambdas_excluded": [0.0, 1.0],
    }
    return {
        "schema_version": "kinofail.unified-reconfirmation-design.v2",
        "scenes": scenes,
        "scene_candidate_streams": streams,
        "scene_selection_contract": deepcopy(external["candidate_source"]),
        "materials": materials,
        "scene_material_assignments": assignments,
        "operators": list(OPERATORS),
        "scale": scale,
        "conflict": deepcopy(external["conflict"]),
        "freshness_contract": {
            "invalid_pilot_exposed_scene_sources_excluded": True,
            "all_material_sources_and_content_hashes_legacy_disjoint": True,
            "all_scale_and_conflict_seed_namespaces_fresh": True,
            "all_operator_interpolation_points_changed_before_generation": True,
            "architecture_features_threshold_and_statistics_inherited_exactly": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    forbidden = [_resolve(value) for value in FORBIDDEN_NEW_DATA_ROOTS]
    if out.exists() or any(path.exists() for path in forbidden):
        raise FileExistsError(
            "F0 must precede every revised asset, scene, corpus, and result"
        )

    design_path = args.design.resolve()
    external = _json(design_path)
    if (
        external.get("schema_version")
        != "kinofail.unified-confirmatory-freeze-input.v1"
        or external.get("status") != "frozen_design_source_before_f0"
        or external.get("confirmatory") is not True
        or external["scale"]["operators"] != OPERATORS
    ):
        raise RuntimeError("invalid external reconfirmation design")

    authority = external["independence_authority"]
    original_f0 = _require_hash(
        _resolve(authority["original_f0_manifest"]), ORIGINAL_F0_SHA256
    )
    _require_hash(
        _resolve(authority["original_f1_manifest"]), ORIGINAL_F1_SHA256
    )
    _require_hash(
        _resolve(authority["original_f2_manifest"]), ORIGINAL_F2_SHA256
    )
    invalid_receipt = _require_hash(
        _resolve(external["invalid_acquisition_pilot"]["receipt"]),
        INVALID_RECEIPT_SHA256,
    )
    qa = _require_hash(
        _resolve(external["acquisition_qualification"]["receipt"]),
        QA_RECEIPT_SHA256,
    )
    archive_path = (
        ROOT
        / "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725"
        / "scene_registry.json"
    )
    archive = _require_hash(archive_path, ARCHIVE_REGISTRY_SHA256)
    if (
        invalid_receipt.get("status") != "invalid_before_model_inference"
        or invalid_receipt["inference_guard"]["checkpoint_loaded"] is not False
        or invalid_receipt["inference_guard"]["prediction_file_created"] is not False
        or qa.get("status") != "passed"
        or qa.get("excluded_from_confirmation") is not True
        or qa.get("scientific_use") != "acquisition_mechanics_only"
    ):
        raise RuntimeError("pilot invalidation or acquisition QA state drift")

    scene_config_path = _resolve(external["candidate_source"]["path"])
    scene_config = _json(scene_config_path)
    exclusions = _json(
        ROOT
        / "configs/data/kinofail_reconfirmation_exposure_exclusions_v2.json"
    )
    reused, exposure_audit = _validate_exposure_and_reuse(
        scene_config=scene_config,
        exclusions=exclusions,
        invalid_receipt=invalid_receipt,
        archive=archive,
    )
    seed_audit = _validate_fresh_seeds(scene_config, archive)
    streams = _candidate_streams(
        reused=reused, scene_config=scene_config, archive=archive
    )
    material_path = _resolve(external["material_contract"]["catalog_path"])
    _materials, material_audit = _validate_materials(material_path)

    old_moderate = set(
        original_f0["design"]["scale"]["seed_rule"]["moderate_lambda_values"]
    )
    old_hard = set(
        original_f0["design"]["scale"]["seed_rule"]["severe_lambda_values"]
    )
    new_moderate = set(external["scale"]["moderate_lambda_points"])
    new_hard = set(external["scale"]["hard_lambda_points"])
    if (
        len(new_moderate) != 8
        or len(new_hard) != 8
        or new_moderate & old_moderate
        or new_hard & old_hard
        or external["scale"]["counterfactual_pairs"] != 10_560
        or external["conflict"]["cases_per_cell"] != 1_500
    ):
        raise RuntimeError("fresh operator-range or count contract drift")

    if external["analysis_contract"]["ordered_primary_gates"] != original_f0[
        "analysis_contract"
    ]["ordered_primary_gates"]:
        raise RuntimeError("primary gates differ from original F0")
    for key in (
        "primary_comparator",
        "one_sided_confidence_level",
        "bootstrap_draws",
        "bootstrap_seed",
        "random_route_seed",
        "independent_units",
        "checkpoint_seeds_are_sensitivity_not_independent_units",
        "score_once_and_report_regardless_of_outcome",
    ):
        if external["analysis_contract"][key] != original_f0[
            "analysis_contract"
        ][key]:
            raise RuntimeError(f"analysis contract differs at {key}")

    checkpoint_audits = _audit_checkpoints(original_f0)
    architecture = deepcopy(original_f0["architecture"])
    architecture["checkpoint_audits"] = checkpoint_audits
    analysis_contract = deepcopy(original_f0["analysis_contract"])
    internal = _internal_design(external=external, streams=streams)
    files = _registered_files(external)
    records = [
        {
            "path": _display(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]

    manifest = {
        "schema_version": "kinofail.unified-reconfirmation-f0.v2",
        "protocol_id": (
            "kinofail-unified-moe-v3-independent-reconfirmation-f0-v2-20260726"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_new_scene_generation",
        "confirmatory": True,
        "new_data_available_at_freeze": False,
        "model_predictions_available_at_freeze": False,
        "architecture": architecture,
        "design": internal,
        "analysis_contract": analysis_contract,
        "external_design": {
            "path": _display(design_path),
            "sha256": _sha256(design_path),
            "schema_version": external["schema_version"],
            "protocol_id": external["protocol_id"],
            "status": external["status"],
        },
        "inherited_contract_hashes": {
            "original_f0_manifest": ORIGINAL_F0_SHA256,
            "architecture_canonical_sha256": _canonical_sha256(architecture),
            "feature_contract_canonical_sha256": _canonical_sha256(
                {
                    "feature_dimensions": architecture["feature_dimensions"],
                    "feature_files": [
                        row
                        for row in records
                        if row["path"]
                        in {
                            "scripts/build_kinofail_unified_invariant_features_v1.py",
                            "scripts/extract_kinofail_realistic_features.py",
                            "kino_vla/eval/realistic_multimodal.py",
                            "kino_vla/eval/c2_temporal_v5.py",
                        }
                    ],
                }
            ),
            "analysis_contract_canonical_sha256": _canonical_sha256(
                analysis_contract
            ),
        },
        "independence_audit": {
            "invalid_pilot": exposure_audit,
            "scene_seed_freshness": seed_audit,
            "material_freshness": material_audit,
            "operator_ranges": {
                "moderate_interval": external["scale"][
                    "moderate_lambda_interval"
                ],
                "hard_interval": external["scale"]["hard_lambda_interval"],
                "all_16_points_disjoint_from_original_f0": True,
            },
            "scale_seed_namespace": {
                "minimum": 2_060_000_000,
                "maximum": 2_060_010_559,
                "prior_repository_collision_search": False,
            },
            "conflict_seed_namespace": {
                "minimum": 2_070_000_000,
                "maximum": 2_070_002_999,
                "prior_repository_collision_search": False,
            },
            "architecture_threshold_features_statistics_changed": False,
            "new_scene_generation_required": 9,
            "unexposed_original_f0_scenes_retained": 21,
            "new_material_source_assets_required": 30,
        },
        "forbidden_new_data_roots": [_display(path) for path in forbidden],
        "frozen_files": records,
    }
    out.mkdir(parents=True, exist_ok=False)
    file_text = "".join(
        f"{row['sha256']}  {row['bytes']}  {row['path']}\n"
        for row in records
    )
    files_path = out / "files.sha256"
    files_path.write_text(file_text, encoding="utf-8")
    manifest["files_manifest_sha256"] = _sha256(files_path)
    manifest_path = out / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "freeze_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  freeze_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "architecture_contract_sha256": manifest[
                    "inherited_contract_hashes"
                ]["architecture_canonical_sha256"],
                "analysis_contract_sha256": manifest[
                    "inherited_contract_hashes"
                ]["analysis_contract_canonical_sha256"],
                "scenes": 30,
                "new_scenes": 9,
                "new_materials": 30,
                "scale_pairs": 10_560,
                "conflict_cases": 3_000,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
