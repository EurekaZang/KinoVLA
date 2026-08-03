#!/usr/bin/env python3
"""Freeze the unified-MoE confirmatory design before any new data exist.

This is the F0 freeze.  It content-locks the five already trained primary
checkpoints, their executable dependencies, the blind inference program, and
the complete sample/seed design.  It deliberately refuses to run after any
confirmatory scene, corpus, feature, prediction, or report root exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.unified_moe import (  # noqa: E402
    EXPERT_NAMES,
    UnifiedEvidenceMoE,
)

EXPECTED_CHECKPOINT_SHA256 = {
    0: "ef357e18cad5ea304a9e14fe053de9aa32628575b2566c3583ac4c572f852054",
    1: "957d146b097429eccb2fad202469ff2dc52843e41caf88a421a8054c95934b3e",
    2: "171298b5dd29f165ff39432c8313d915802d3aa672f72c756a785a693f78605a",
    3: "9c7cc2bd2dc2a1270a679180c3706f4046086a0fd367b565d004499ecc41dcd4",
    4: "4646d0683ed0958ceffff41868638dbef9743ad5db41720a9583136796fe40fd",
}
EXPECTED_CLASSES = [
    "adhesion",
    "compliant_terrain",
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "nominal",
    "obs_bias",
    "overload",
    "region_collapse",
]
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
DOMAINS = ("life", "production", "wild")
ROUTE_THRESHOLD = 0.8
SCENES_PER_DOMAIN = 10
MATERIALS_PER_DOMAIN = 10
MATERIALS_PER_SCENE = 2
SCALE_SEEDS_PER_SCENE_MATERIAL_OPERATOR = 16
CONFLICT_CASES_PER_CELL_SCENE_MATERIAL = 25
SCALE_SEED_BASE = 2_030_000_000
CONFLICT_SEED_BASE = 2_040_000_000
SCENE_CANDIDATES_PATH = (
    ROOT
    / "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json"
)
DEFAULT_CONFIRMATORY_ROOTS = (
    "outputs/kinofail_confirmatory_v1",
    "outputs/kinofail_realistic/scenes_unified_moe_v3_confirmatory_v1",
    "outputs/kinofail_realistic/corpus_unified_moe_v3_confirmatory_v1",
    "outputs/eval/unified_moe_v3_confirmatory_v1",
    "outputs/assets/terrain_pbr_confirmatory_v1",
    "outputs/assets/forest_hdri_polyhaven_confirmatory_v1",
)
DEFAULT_DESIGN_PATH = ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
MANDATORY_PIPELINE_FILES = (
    "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json",
    "configs/data/kinofail_unified_confirmatory_hdri_v1.json",
    "scripts/run_kinofail_confirmatory_indoor_scene_v1.py",
    "scripts/run_kinofail_confirmatory_wild_scene_v1.py",
    "scripts/run_embodiedgen_go2_scene_qa_failure_closed_v1.py",
    "scripts/build_kinofail_confirmatory_scene_registry_v1.py",
    "scripts/build_kinofail_unified_confirmatory_v1_schedules.py",
    "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py",
    "scripts/run_kinofail_confirmatory_scale_shard_v1.py",
    "scripts/seal_and_prune_kinofail_confirmatory_shard_v1.py",
    "scripts/seal_kinofail_confirmatory_f1.py",
    "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
    "scripts/extract_kinofail_confirmatory_t2_features_v1.py",
    "scripts/merge_kinofail_confirmatory_t2_features_v1.py",
    "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
    "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
    "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
    "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
    "scripts/extract_kinofail_realistic_c1_causal_features_v1.py",
    "scripts/build_kinofail_realistic_c2_v5_features.py",
    "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def checkpoint_paths() -> dict[int, Path]:
    root = ROOT / "outputs/eval/unified_moe_v3_development_full/checkpoints"
    return {
        seed: root / f"seed{seed}/unified_moe.pkl" for seed in sorted(EXPECTED_CHECKPOINT_SHA256)
    }


def confirmatory_design() -> dict[str, Any]:
    candidate_config = json.loads(
        SCENE_CANDIDATES_PATH.read_text(encoding="utf-8")
    )
    scenes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    global_index = 0
    for domain in DOMAINS:
        for local_index in range(SCENES_PER_DOMAIN):
            scenes.append(
                {
                    "scene_index": global_index,
                    "scene_id": (f"confirm_v1_{domain}_scene_{local_index:02d}"),
                    "domain": domain,
                    "admission_slot": local_index,
                    "source_selection": (
                        "first_ten_model_blind_admissions"
                        if domain in {"life", "production"}
                        else "fixed_wild_candidate"
                    ),
                }
            )
            materials.append(
                {
                    "material_index": global_index,
                    "material_id": (f"confirm_v1_{domain}_pbr_{local_index:02d}"),
                    "domain": domain,
                    "must_be_new_source_asset": True,
                }
            )
            global_index += 1

    scene_material_assignments: list[dict[str, Any]] = []
    for scene in scenes:
        domain_start = DOMAINS.index(scene["domain"]) * SCENES_PER_DOMAIN
        local_index = int(scene["scene_index"]) - domain_start
        material_indices = (
            domain_start + local_index,
            domain_start + ((local_index + 5) % MATERIALS_PER_DOMAIN),
        )
        for slot, material_index in enumerate(material_indices):
            scene_material_assignments.append(
                {
                    "context_index": len(scene_material_assignments),
                    "scene_id": scene["scene_id"],
                    "material_id": materials[material_index]["material_id"],
                    "material_slot": slot,
                    "domain": scene["domain"],
                }
            )

    scale_seed_rule = {
        "formula": (
            "2030000000 + "
            "(((scene_index*2 + material_slot)*11 + operator_index)*16 "
            "+ replicate_index)"
        ),
        "replicate_indices": list(range(16)),
        "moderate_replicate_indices": list(range(8)),
        "severe_replicate_indices": list(range(8, 16)),
        "moderate_lambda_values": [0.15 + (index + 0.5) * (0.45 - 0.15) / 8 for index in range(8)],
        "severe_lambda_values": [0.65 + (index + 0.5) * (0.95 - 0.65) / 8 for index in range(8)],
        "operator_interpolation": "p(lambda)=(1-lambda)*moderate+lambda*severe",
        "old_endpoint_lambdas_excluded": [0.0, 1.0],
    }
    scale_pairs = (
        len(scenes) * MATERIALS_PER_SCENE * len(OPERATORS) * SCALE_SEEDS_PER_SCENE_MATERIAL_OPERATOR
    )
    cases_per_cell = len(scenes) * MATERIALS_PER_SCENE * CONFLICT_CASES_PER_CELL_SCENE_MATERIAL
    candidate_streams = {
        domain: [
            dict(row)
            for row in (
                candidate_config["indoor_candidates"]
                if domain in {"life", "production"}
                else candidate_config["wild_scenes"]
            )
            if str(row["domain"]) == domain
        ]
        for domain in DOMAINS
    }
    if (
        len(candidate_streams["life"]) != 20
        or len(candidate_streams["production"]) != 20
        or len(candidate_streams["wild"]) != 10
    ):
        raise RuntimeError("confirmatory scene candidate stream drift")
    return {
        "schema_version": "kinofail.unified-confirmatory-design.v1",
        "scenes": scenes,
        "scene_candidate_streams": candidate_streams,
        "scene_selection_contract": {
            "life": "first ten passed candidates in the frozen list",
            "production": "first ten passed candidates in the frozen list",
            "wild": "all ten fixed candidates must pass",
            "selection_uses_model_predictions": False,
            "failed_candidates_remain_in_attrition_ledger": True,
            "replacement_or_repair": False,
        },
        "materials": materials,
        "scene_material_assignments": scene_material_assignments,
        "operators": OPERATORS,
        "scale": {
            "counterfactual_pairs": scale_pairs,
            "physical_episodes": 2 * scale_pairs,
            "appearance_views_per_episode": 3,
            "model_rows": 6 * scale_pairs,
            "fresh_seeds_per_scene_material_operator": (SCALE_SEEDS_PER_SCENE_MATERIAL_OPERATOR),
            "seed_rule": scale_seed_rule,
        },
        "conflict": {
            "cells": ["T2_vision_decisive", "T3_proprio_decisive"],
            "cases_per_cell": cases_per_cell,
            "cases_total": 2 * cases_per_cell,
            "cases_per_cell_scene_material": (CONFLICT_CASES_PER_CELL_SCENE_MATERIAL),
            "records_per_case": 6,
            "model_rows": 12 * cases_per_cell,
            "seed_formula": ("2040000000 + cell_index*1500 + context_index*25 + local_case_index"),
        },
        "freshness_contract": {
            "shared_new_scenes_across_scale_and_conflict": True,
            "shared_new_materials_across_scale_and_conflict": True,
            "all_scene_ids_and_source_seeds_absent_from_prior_corpora": True,
            "all_material_source_ids_and_content_hashes_absent_from_prior_assets": True,
            "all_physical_and_case_seeds_absent_from_prior_schedules": True,
            "all_operator_parameter_vectors_absent_from_prior_schedules": True,
        },
    }


def _route_contract_checks(model: UnifiedEvidenceMoE) -> dict[str, bool]:
    probabilities = np.zeros((1, len(EXPERT_NAMES), len(model.classes)))
    probabilities[0, EXPERT_NAMES.index("proprio"), 0] = 1.0
    probabilities[0, EXPERT_NAMES.index("vision"), 1] = 1.0
    probabilities[0, EXPERT_NAMES.index("joint"), 2] = 1.0
    thresholds = {"vision": ROUTE_THRESHOLD, "joint": ROUTE_THRESHOLD}
    exact = model._select_routes(
        probabilities,
        np.asarray([[ROUTE_THRESHOLD, 0.0]]),
        thresholds,
    )
    above = model._select_routes(
        probabilities,
        np.asarray([[np.nextafter(ROUTE_THRESHOLD, np.inf), 0.0]]),
        thresholds,
    )
    tie = model._select_routes(
        probabilities,
        np.asarray([[0.9, 0.9]]),
        thresholds,
    )
    return {
        "score_equal_0_8_keeps_proprio": bool(exact[0] == EXPERT_NAMES.index("proprio")),
        "score_strictly_above_0_8_selects_vision": bool(above[0] == EXPERT_NAMES.index("vision")),
        "equal_vision_joint_excess_selects_vision": bool(tie[0] == EXPERT_NAMES.index("vision")),
    }


def audit_checkpoint(path: Path, seed: int) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    expected_hash = EXPECTED_CHECKPOINT_SHA256[seed]
    if actual_hash != expected_hash:
        raise RuntimeError(f"checkpoint seed{seed} hash mismatch: {actual_hash}")
    with path.open("rb") as stream:
        model = pickle.load(stream)  # noqa: S301 - hash-pinned local artifact
    if not isinstance(model, UnifiedEvidenceMoE):
        raise TypeError(f"unexpected checkpoint type: {path}")
    if list(model.classes) != EXPECTED_CLASSES:
        raise RuntimeError(f"class contract mismatch: {path}")
    if model.route_thresholds != {
        "vision": ROUTE_THRESHOLD,
        "joint": ROUTE_THRESHOLD,
    }:
        raise RuntimeError(f"route threshold mismatch: {path}")
    for name, expert in model.experts.items():
        estimator = expert.model
        if (
            name not in EXPERT_NAMES
            or estimator.n_estimators != 192
            or estimator.min_samples_leaf != 2
        ):
            raise RuntimeError(f"expert architecture mismatch: {path}")
    for name, router in model.routers.items():
        if (
            name not in ("vision", "joint")
            or getattr(router, "n_estimators", None) != 256
            or getattr(router, "min_samples_leaf", None) != 6
        ):
            raise RuntimeError(f"router architecture mismatch: {path}")
    route_checks = _route_contract_checks(model)
    if not all(route_checks.values()):
        raise RuntimeError(f"route semantics mismatch: {route_checks}")
    return {
        "seed": seed,
        "path": _display_path(path),
        "sha256": actual_hash,
        "classes": list(model.classes),
        "route_thresholds": dict(model.route_thresholds),
        "route_contract_checks": route_checks,
        "fit_audit": model.fit_audit,
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _design_frozen_entries(
    value: Any,
) -> list[tuple[str, str | None]]:
    """Recursively accept list, named-object, and path-to-hash registries."""

    entries: list[tuple[str, str | None]] = []
    if isinstance(value, list):
        for child in value:
            entries.extend(_design_frozen_entries(child))
        return entries
    if isinstance(value, str):
        entries.append((value, None))
        return entries
    if not isinstance(value, dict):
        return entries
    path = value.get("path")
    if isinstance(path, str):
        expected = value.get("sha256")
        if expected is not None and not _is_sha256(expected):
            raise ValueError(f"invalid frozen-file SHA-256 for {path}")
        entries.append((path, expected))
        return entries
    for key, child in value.items():
        if (
            isinstance(key, str)
            and ("/" in key or key.endswith((".py", ".json", ".yaml", ".yml")))
            and _is_sha256(child)
        ):
            entries.append((key, str(child)))
        else:
            entries.extend(_design_frozen_entries(child))
    return entries


def design_frozen_files(
    design_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    design_path = design_path.resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise TypeError(f"confirmatory design must be an object: {design_path}")
    if "frozen_files" not in design:
        raise RuntimeError("confirmatory design must recursively register frozen_files")
    entries = _design_frozen_entries(design["frozen_files"])
    if not entries:
        raise RuntimeError("confirmatory design frozen_files is empty")
    paths: list[Path] = []
    for raw_path, expected_hash in entries:
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"design-registered frozen file is missing: {resolved}")
        if expected_hash is not None and sha256_file(resolved) != expected_hash:
            raise RuntimeError(f"design-registered frozen file hash mismatch: {resolved}")
        paths.append(resolved)
    return design, paths


def validate_external_design_contract(
    external: dict[str, Any],
    internal: dict[str, Any],
) -> dict[str, bool]:
    """Prove the human-readable authority and executable F0 agree exactly."""

    candidate = external.get("candidate_source", {})
    material = external.get("material_contract", {})
    scale = external.get("scale", {})
    conflict = external.get("conflict", {})
    checks = {
        "schema": (
            external.get("schema_version")
            == "kinofail.unified-confirmatory-freeze-input.v1"
        ),
        "status": external.get("status") == "frozen_design_source_before_f0",
        "confirmatory": external.get("confirmatory") is True,
        "candidate_authority": (
            candidate.get("path")
            == "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json"
            and candidate.get("authoritative") is True
            and candidate.get("life_candidates") == 20
            and candidate.get("production_candidates") == 20
            and candidate.get("wild_fixed_scenes") == 10
            and candidate.get("selection", {}).get("replacement_or_repair") is False
            and candidate.get("selection", {}).get(
                "model_or_anomaly_outcomes_visible_during_selection"
            )
            is False
        ),
        "material_contract": (
            material.get("catalog_path")
            == "configs/data/kinofail_confirmatory_terrain_pbr_v1.yaml"
            and material.get("count") == 30
            and material.get("per_domain") == 10
            and material.get("contexts_per_scene") == 2
        ),
        "scale_contract": (
            scale.get("operators") == internal["operators"]
            and scale.get("scene_count") == len(internal["scenes"])
            and scale.get("material_contexts_per_scene") == 2
            and scale.get("replicates_per_scene_material_operator") == 16
            and scale.get("counterfactual_pairs")
            == internal["scale"]["counterfactual_pairs"]
            and scale.get("physical_episodes")
            == internal["scale"]["physical_episodes"]
            and scale.get("seed_formula")
            == internal["scale"]["seed_rule"]["formula"].replace(" ", "")
        ),
        "conflict_contract": (
            conflict.get("cells") == internal["conflict"]["cells"]
            and conflict.get("cases_per_scene_material_cell") == 25
            and conflict.get("cases_per_cell")
            == internal["conflict"]["cases_per_cell"]
            and conflict.get("cases_total")
            == internal["conflict"]["cases_total"]
            and conflict.get("records_per_case") == 6
            and str(conflict.get("seed_formula", "")).replace(" ", "")
            == str(internal["conflict"]["seed_formula"]).replace(" ", "")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"external and executable confirmatory designs disagree: {checks}")
    return checks


def required_frozen_files(
    design_path: Path,
    design_registered_paths: list[Path],
) -> list[Path]:
    paths = [
        design_path.resolve(),
        ROOT / "kino_vla/eval/unified_moe.py",
        ROOT / "kino_vla/eval/realistic_multimodal.py",
        ROOT / "kino_vla/eval/c2_temporal_v5.py",
        ROOT / "kino_vla/map/clip_appearance.py",
        ROOT / "kino_vla/data/runtime_manifest.py",
        ROOT / "scripts/build_kinofail_unified_invariant_features_v1.py",
        ROOT / "scripts/extract_kinofail_realistic_features.py",
        ROOT / "scripts/run_frozen_stage_with_audit_v1.py",
        ROOT / "scripts/freeze_kinofail_unified_confirmatory_f0.py",
        ROOT / "scripts/audit_kinofail_confirmatory_freshness_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "configs/eval/kinofail_unified_moe_v1_nomargin_development.json",
        ROOT / "outputs/eval/unified_moe_v3_development_full/report.json",
        ROOT / "outputs/eval/unified_moe_v3_publication_audit/report.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        Path(
            "/home/eureka/.cache/huggingface/hub/"
            "models--openai--clip-vit-base-patch32/snapshots/"
            "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/"
            "pytorch_model.bin"
        ),
    ]
    paths.extend(ROOT / value for value in MANDATORY_PIPELINE_FILES)
    paths.extend(design_registered_paths)
    paths.extend(checkpoint_paths().values())
    return paths


def _fail_if_any_exists(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "F0 must precede every confirmatory artifact; already exists: " + ", ".join(existing)
        )


def _file_manifest(paths: list[Path]) -> tuple[list[dict[str, Any]], str]:
    records = []
    for path in sorted({item.resolve() for item in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": _display_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    text = "".join(f"{row['sha256']}  {row['bytes']}  {row['path']}\n" for row in records)
    return records, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        type=Path,
        default=DEFAULT_DESIGN_PATH,
        help=("Pre-data design whose recursively nested frozen_files registry is authoritative."),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(ROOT / "outputs/freeze/unified_moe_v3_confirmatory_f0"),
    )
    parser.add_argument(
        "--forbid-existing",
        type=Path,
        action="append",
        default=[],
        help="Additional path that proves new-data work already started.",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite F0 freeze: {out}")
    forbidden = [(ROOT / value).resolve() for value in DEFAULT_CONFIRMATORY_ROOTS]
    forbidden.extend(path.resolve() for path in args.forbid_existing)
    _fail_if_any_exists(forbidden)

    design_path = args.design.resolve()
    external_design, design_registered_paths = design_frozen_files(design_path)
    checkpoint_audits = [audit_checkpoint(path, seed) for seed, path in checkpoint_paths().items()]
    file_records, file_text = _file_manifest(
        required_frozen_files(design_path, design_registered_paths)
    )
    design = confirmatory_design()
    external_contract_checks = validate_external_design_contract(
        external_design, design
    )
    if design["scale"]["counterfactual_pairs"] != 10_560:
        raise RuntimeError("scale design count drift")
    if design["conflict"]["cases_per_cell"] != 1_500:
        raise RuntimeError("conflict design count drift")

    manifest = {
        "schema_version": "kinofail.unified-confirmatory-f0.v6",
        "protocol_id": "kinofail-unified-moe-v3-confirmatory-f0-v6-20260725",
        "status": "frozen_before_new_scene_generation",
        "created_utc": datetime.now(UTC).isoformat(),
        "confirmatory": True,
        "new_data_available_at_freeze": False,
        "model_predictions_available_at_freeze": False,
        "supersedes_invalid_f0": {
            "manifest": "outputs/freeze/unified_moe_v3_confirmatory_f0_v5/freeze_manifest.json",
            "reason": (
                "the indoor admission aggregator incorrectly required the "
                "identity-only source manifest to expose an audit passed field"
            ),
            "successful_indoor_scenes_before_v6": 0,
            "successful_wild_scenes_before_v6": 6,
            "anomaly_episodes_before_v6": 0,
            "model_inferences_before_v6": 0,
            "archived_attempts": (
                "outputs/kinofail_confirmatory_invalid_f0_v5_admission_aggregation_20260725"
            ),
            "scientific_design_changed": False,
            "execution_only_correction": (
                "require source manifest presence and hash evidence while "
                "requiring passed=true only from the eight actual audit files"
            )
        },
        "architecture": {
            "checkpoint_count": 5,
            "checkpoint_audits": checkpoint_audits,
            "default_expert": "proprio",
            "override_experts_in_priority_order": ["vision", "joint"],
            "route_score_threshold": ROUTE_THRESHOLD,
            "threshold_comparison": "strictly_greater_than",
            "equal_excess_tie_break": "vision",
            "fit_or_refit_permitted": False,
            "feature_dimensions": {"visual": 1536, "proprio": 80},
        },
        "design": design,
        "external_design": {
            "path": _display_path(design_path),
            "sha256": sha256_file(design_path),
            "schema_version": external_design.get("schema_version"),
            "protocol_id": external_design.get("protocol_id"),
            "status": external_design.get("status"),
            "recursively_registered_frozen_files": len(set(design_registered_paths)),
            "contract_checks": external_contract_checks,
        },
        "analysis_contract": {
            "ordered_primary_gates": [
                (
                    "scale and conflict battery-wise non-inferiority to "
                    "late-average at margin -0.01"
                ),
                (
                    "worst-battery balanced-accuracy superiority to "
                    "late-average at margin +0.005"
                ),
                (
                    "equal-battery macro balanced-accuracy superiority to "
                    "late-average at margin 0"
                ),
            ],
            "primary_comparator": "late_average",
            "one_sided_confidence_level": 0.975,
            "secondary_metrics": [
                "worst-battery balanced accuracy",
                "75%-coverage strict-group selective risk",
                "T2/T3 decision-critical route fidelity",
                "strict all-checkpoint pair/case correctness",
                "scene/material/operator-condition balanced accuracy",
            ],
            "independent_units": {
                "scale": "counterfactual pair",
                "conflict": "matched physical case",
                "scene": "scene geometry",
                "material": "PBR source asset",
            },
            "checkpoint_seeds_are_sensitivity_not_independent_units": True,
            "coverage_grid": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
            "bootstrap_draws": 20_000,
            "bootstrap_seed": 2_027_012_751,
            "random_route_seed": 2_027_012_790,
            "score_once_and_report_regardless_of_outcome": True,
        },
        "forbidden_confirmatory_roots": [_display_path(path) for path in forbidden],
        "frozen_files": file_records,
    }
    out.mkdir(parents=True, exist_ok=False)
    files_path = out / "files.sha256"
    files_path.write_text(file_text, encoding="utf-8")
    manifest["files_manifest_sha256"] = sha256_file(files_path)
    manifest_path = out / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = out / "freeze_manifest.sha256"
    sidecar.write_text(
        f"{sha256_file(manifest_path)}  freeze_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "scale_pairs": design["scale"]["counterfactual_pairs"],
                "conflict_cases_per_cell": (design["conflict"]["cases_per_cell"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
