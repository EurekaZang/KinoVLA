#!/usr/bin/env python3
"""Freeze C2 v5 T3 Amendment 1 after a near-threshold swap QA failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    base_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v5_t3"
    )
    summaries = list((base_corpus / "pair_summaries").glob("*.json"))
    passed_summaries = [
        path for path in summaries if _json(path).get("passed") is True
    ]
    failed_summaries = [
        path for path in summaries if _json(path).get("passed") is not True
    ]
    manifests = list(base_corpus.rglob("manifest.json"))
    failed_manifests = [
        path
        for path in manifests
        if _json(path).get("runtime_validation", {}).get("passed")
        is not True
    ]
    if (
        len(summaries) != 11
        or len(passed_summaries) != 10
        or len(failed_summaries) != 1
        or len(manifests) != 22
        or len(failed_manifests) != 1
        or _json(failed_manifests[0])
        .get("runtime_validation", {})
        .get("issues")
        != ["rgb_view_swap_02_appearance_effect_too_small"]
    ):
        raise RuntimeError("unexpected base T3 collection footprint")
    measured = _json(failed_manifests[0])["runtime_validation"][
        "measured"
    ]["mean_appearance_pair_rgb_l1"]["swap_02"]
    threshold = _json(failed_manifests[0])["runtime_validation"][
        "thresholds"
    ]["min_mean_appearance_pair_rgb_l1"]
    if not (float(measured) < float(threshold)):
        raise RuntimeError("unexpected swap QA diagnosis")
    forbidden = (
        ROOT / "outputs/eval/c2_bidirectional_v5/formal_features",
        ROOT / "outputs/eval/c2_bidirectional_v5/formal",
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v5_t3_amendment1",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError(
            "amendment must precede v5 features, predictions, and recollection"
        )

    design = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v5_t3_amendment1"
    )
    schedule = design / "schedule.jsonl"
    cases = design / "case_schedule.jsonl"
    audit_path = design / "audit.json"
    audit = _json(audit_path)
    rows = _jsonl(schedule)
    if (
        audit.get("passed") is not True
        or audit.get("design_tag")
        != "c2_bidirectional_confirmation_v5_t3_a1_strong_swaps"
        or _sha(schedule) != audit["schedule_sha256"]
        or _sha(cases) != audit["case_schedule_sha256"]
        or len(rows) != 60
    ):
        raise RuntimeError("invalid amended T3 design")
    for row in rows:
        views = {
            str(view["appearance_view_id"]): view
            for view in row["appearance_views"]
        }
        if (
            views["primary"]["material_asset_id"]
            == views["swap_01"]["material_asset_id"]
            or views["primary"]["material_asset_id"]
            == views["swap_02"]["material_asset_id"]
        ):
            raise RuntimeError("amended T3 swap is not materially distinct")

    base_t3_protocol_path = (
        ROOT
        / "configs/eval/kinofail_realistic_c2_v5_t3_collection.json"
    )
    base_evaluation_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v5.json"
    )
    t2_protocol_path = (
        ROOT
        / "configs/eval/kinofail_realistic_c2_v5_t2_collection.json"
    )
    amendment_path = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v5_t3_amendment1.json"
    )
    t3_protocol_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v5_t3_collection_amendment1.json"
    )
    evaluation_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v5_amendment1.json"
    )
    if any(
        path.exists()
        for path in (
            amendment_path,
            t3_protocol_path,
            evaluation_path,
        )
    ):
        raise FileExistsError("C2 v5 Amendment 1 is already frozen")

    amendment = {
        "schema_version": (
            "kinofail.realistic-c2-v5-administrative-amendment.v1"
        ),
        "amendment_id": "C2-v5-T3-A1-strong-material-swaps",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "frozen_after_collection_qa_before_features_or_predictions"
        ),
        "reason": (
            "The base schedule reused the primary material asset for "
            "swap_02 and changed only surface-state parameters. One life "
            "episode measured mean aligned RGB L1 "
            f"{float(measured):.6f}, just below the unchanged "
            f"{float(threshold):.3f} QA gate. Rather than lower the gate, "
            "both nuisance views are now required to use a material asset "
            "distinct from the primary."
        ),
        "change": {
            "scope": "T3 appearance-intervention schedule only",
            "primary_material_distinct_from_swap_01": True,
            "primary_material_distinct_from_swap_02": True,
            "appearance_qa_threshold_changed": False,
            "all_thirty_pairs_recollected_from_scratch": True,
        },
        "unchanged": [
            "C2 model architecture and hyperparameters",
            "post-interaction temporal contract",
            "six-scene development data and report",
            "three confirmation scene instances",
            "T2 schedule and collection",
            "T3 operators and physical parameters",
            "severity pattern",
            "acceptance thresholds",
            "bootstrap seed and draws",
        ],
        "outcome_footprint_at_amendment": {
            "base_t3_complete_pairs": len(summaries),
            "base_t3_passed_pairs": len(passed_summaries),
            "base_t3_failed_pairs": len(failed_summaries),
            "formal_feature_rows": 0,
            "formal_predictions": 0,
            "model_predictions_inspected": False,
        },
        "disposition": {
            "base_t3_corpus": (
                "retained as collection-QA audit only; excluded from "
                "formal evaluation"
            ),
            "publication_t3_corpus": (
                "all 30 pairs recollected under Amendment 1"
            ),
        },
        "passed": True,
    }
    _write(amendment_path, amendment)

    registry = (
        ROOT
        / "configs/data"
        / "kinofail_realistic_c2_v5_confirmation_scene_registry.json"
    )
    collector = (
        ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v3.py"
    )
    runner = (
        ROOT / "scripts/run_kinofail_realistic_c2_v4_t3_isolated.py"
    )
    runtime_manifest = ROOT / "kino_vla/data/runtime_manifest.py"
    group_ids = sorted(
        {str(row["counterfactual_group_id"]) for row in rows}
    )
    t3_protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": (
            "kinofail-realistic-c2-v5-t3-collection-amendment1"
        ),
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": audit["design_tag"],
        "scope": (
            "60 physical episodes / 30 nominal-anomaly O7/O8 pairs "
            "with two strong material swaps on three fresh C2 scenes."
        ),
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "runner_path": str(runner.relative_to(ROOT)),
        "runner_sha256": _sha(runner),
        "runtime_manifest_path": str(
            runtime_manifest.relative_to(ROOT)
        ),
        "runtime_manifest_sha256": _sha(runtime_manifest),
        "design_audit_sha256": _sha(audit_path),
        "case_schedule_sha256": _sha(cases),
        "model_predictions_available_at_freeze": False,
        "allowed": {
            "counterfactual_group_ids": group_ids,
            "target_operators": [
                "O7_visual_remap",
                "O8_invisible_collider",
            ],
            "scene_families": sorted(
                {str(row["scene_family"]) for row in rows}
            ),
            "physical_realizations": sorted(
                {str(row["physical_realization"]) for row in rows}
            ),
            "geometry_profiles": sorted(
                {str(row["geometry_profile"]) for row in rows}
            ),
            "severity_ids": ["moderate", "severe"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 60,
            "counterfactual_pairs": 30,
            "appearance_views_per_episode": 3,
            "strong_material_swaps_per_episode": 2,
            "minimum_mean_appearance_pair_rgb_l1": 0.015,
            "scene_families": 3,
            "domains": 3,
            "operators": 2,
            "formal_collection_status": "formal_pilot",
            "runtime_schema": "kinofail.realistic-runtime.v5",
            "outcome_strength_is_reported_not_gated": True,
            "per_scene_parameter_tuning_forbidden": True,
            "isaac_processes": 30,
            "pairs_per_process": 1,
            "maximum_frame_highlight_fraction": 0.995,
            "maximum_mean_highlight_fraction": 0.45,
            "a8_in_scope": False,
        },
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": _sha(amendment_path),
        "supersedes_sha256": _sha(base_t3_protocol_path),
    }
    _write(t3_protocol_path, t3_protocol)

    evaluation = _json(base_evaluation_path)
    evaluation.update(
        {
            "protocol_id": (
                "kinofail-realistic-c2-bidirectional-"
                "confirmation-v5-amendment1"
            ),
            "status": (
                "frozen_amendment1_before_full_t3_recollection"
            ),
            "created_utc": datetime.now(UTC).isoformat(),
            "t3_schedule_sha256": _sha(schedule),
            "t2_collection_protocol_sha256": _sha(
                t2_protocol_path
            ),
            "t3_collection_protocol_sha256": _sha(
                t3_protocol_path
            ),
            "amendment_model_predictions_available_at_freeze": False,
            "administrative_amendment": {
                "amendment_id": amendment["amendment_id"],
                "amendment_sha256": _sha(amendment_path),
                "architecture_changed": False,
                "temporal_contract_changed": False,
                "acceptance_thresholds_changed": False,
                "model_predictions_inspected": False,
                "all_t3_pairs_recollected": True,
            },
            "base_protocol_path": str(
                base_evaluation_path.relative_to(ROOT)
            ),
            "base_protocol_sha256": _sha(base_evaluation_path),
        }
    )
    _write(evaluation_path, evaluation)
    print(
        json.dumps(
            {
                "passed": True,
                "amendment": str(amendment_path),
                "amendment_sha256": _sha(amendment_path),
                "t3_protocol": str(t3_protocol_path),
                "t3_protocol_sha256": _sha(t3_protocol_path),
                "evaluation_protocol": str(evaluation_path),
                "evaluation_protocol_sha256": _sha(evaluation_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
