#!/usr/bin/env python3
"""Freeze the realistic direct-C4 schedule before any direct action outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_OPERATORS = {
    "O2_compliance",
    "O4_tether",
    "O5_payload",
    "O8_invisible_collider",
    "O9_high_centering",
}
TARGET_ATTRIBUTION = "adhesion"
RELEASE_THRESHOLD = 0.5
REQUIRED_SEED_AGREEMENT = 5


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return 300_000_000 + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_700_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-schedule",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl",
    )
    parser.add_argument(
        "--snapshot-records",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/snapshots/snapshot_records.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/features/feature_manifest.json",
    )
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--actor",
        type=Path,
        default=ROOT / "outputs/locomotion/recovery_route_v1/policy.pt",
    )
    parser.add_argument(
        "--collector",
        type=Path,
        default=ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v1.py",
    )
    parser.add_argument(
        "--schedule-out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_c4_direct_v1/schedule.jsonl",
    )
    parser.add_argument(
        "--protocol-out",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_c4_direct_formal_v1.json",
    )
    args = parser.parse_args()

    resolved = {
        key: value.resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    if resolved["schedule_out"].exists() or resolved["protocol_out"].exists():
        raise FileExistsError(
            "C4 freeze already exists; refusing a post-outcome overwrite"
        )

    registry = _json(resolved["scene_registry"])
    test_scene_rows = [
        row for row in registry["scenes"] if str(row["split"]) == "test"
    ]
    test_scenes = {str(row["scene_id"]): str(row["domain"]) for row in test_scene_rows}
    if len(test_scenes) != 3 or set(test_scenes.values()) != {
        "life",
        "production",
        "wild",
    }:
        raise RuntimeError("C4 requires one held-out scene from each of three domains")

    source_by_episode = {
        str(row["episode_id"]): row for row in _jsonl(resolved["source_schedule"])
    }
    records = [
        row
        for row in _jsonl(resolved["snapshot_records"])
        if str(row["scene_family"]) in test_scenes
        and str(row["target_operator"]) in TARGET_OPERATORS
        and str(row["condition"]) == "anomaly"
        and str(row["severity_id"]) == "severe"
        and str(row["appearance_intervention_id"]) == "primary"
    ]
    # Selection is deliberately independent of model output and test truth.
    expected_cells = Counter(
        (str(row["scene_family"]), str(row["target_operator"])) for row in records
    )
    if (
        len(records) != 75
        or set(expected_cells.values()) != {5}
        or len(expected_cells) != 15
    ):
        raise RuntimeError(
            f"expected 3 scenes x 5 operators x 5 nuisances, got {expected_cells}"
        )

    prediction_rows = [
        row
        for row in _jsonl(resolved["predictions"])
        if str(row["axis"]) == "scene_and_material"
        and str(row["method"]) == "structured_bidirectional"
        and bool(row["headline_primary_view"])
    ]
    predictions_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        predictions_by_sample[str(row["sample_id"])].append(row)

    checkpoint_root = (
        resolved["predictions"].parent / "checkpoints" / "scene_and_material"
    )
    checkpoint_paths = [
        checkpoint_root / f"seed{seed}" / "structured_bidirectional.pkl"
        for seed in range(5)
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    schedule: list[dict[str, Any]] = []
    release_counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda row: str(row["sample_id"])):
        sample_id = str(record["sample_id"])
        model_rows = predictions_by_sample[sample_id]
        if len(model_rows) != 5 or {int(row["seed"]) for row in model_rows} != set(
            range(5)
        ):
            raise RuntimeError(f"incomplete five-seed prediction: {sample_id}")
        classes = [str(value) for value in model_rows[0]["classes"]]
        if any([str(value) for value in row["classes"]] != classes for row in model_rows):
            raise RuntimeError(f"class order mismatch: {sample_id}")
        probabilities = np.asarray(
            [row["probabilities"] for row in model_rows], dtype=np.float64
        )
        mean_probability = probabilities.mean(axis=0)
        prediction = classes[int(np.argmax(mean_probability))]
        seed_predictions = [str(row["prediction"]) for row in model_rows]
        agreement = int(sum(value == prediction for value in seed_predictions))
        adhesion_probability = float(
            mean_probability[classes.index(TARGET_ATTRIBUTION)]
        )
        release = (
            prediction == TARGET_ATTRIBUTION
            and agreement >= REQUIRED_SEED_AGREEMENT
            and adhesion_probability >= RELEASE_THRESHOLD
        )
        release_counts[
            f"{record['scene_family']}|{record['target_operator']}|{release}"
        ] += 1

        episode_id = str(record["physical_episode_id"])
        if episode_id not in source_by_episode:
            raise RuntimeError(f"source schedule row missing: {episode_id}")
        source_record = source_by_episode[episode_id]
        if str(source_record["condition"]) != "anomaly":
            raise RuntimeError("C4 source record must be anomalous")
        decision = {
            "schema_version": "kinofail.realistic-c4-frozen-decision.v1",
            "policy": "structured_bidirectional_5seed_ensemble",
            "input_fields": ["rgb_window", "proprio_window"],
            "source_sample_id": sample_id,
            "source_snapshot_record_sha256": hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "predicted_attribution": prediction,
            "classes": classes,
            "mean_probabilities": mean_probability.tolist(),
            "seed_predictions": seed_predictions,
            "seed_agreement": agreement,
            "release_target": TARGET_ATTRIBUTION,
            "release_probability_threshold": RELEASE_THRESHOLD,
            "minimum_seed_agreement": REQUIRED_SEED_AGREEMENT,
            "release": bool(release),
            "recovery_action": "backstep_release",
            "safe_action": "hold_and_request",
        }
        schedule.append(
            {
                "schema_version": "kinofail.realistic-c4-direct-schedule.v1",
                "case_id": f"{sample_id}__direct_c4",
                "scene_cluster": str(record["scene_family"]),
                "domain": str(record["domain"]),
                "operator": str(record["target_operator"]),
                "truth_attribution": str(record["attribution_category"]),
                "source_sample_id": sample_id,
                "source_physical_episode_id": episode_id,
                "source_counterfactual_group_id": str(
                    record["counterfactual_group_id"]
                ),
                "source_record": source_record,
                "reset_seed": _stable_seed("realistic-c4-direct-v1", sample_id),
                "decision": decision,
                "branch_order": ["selective", "always_safe"],
                "selection_rule": (
                    "all severe primary-view anomaly snapshots for the five "
                    "predeclared action operators in all three registry test scenes"
                ),
            }
        )

    # The protocol may inspect coverage for a freeze audit, but it may not adapt
    # threshold/case selection because no direct action outcome exists yet.
    release_count = sum(bool(row["decision"]["release"]) for row in schedule)
    if release_count == 0:
        raise RuntimeError("frozen rule has zero release coverage")

    schedule_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in schedule
    )
    resolved["schedule_out"].parent.mkdir(parents=True, exist_ok=True)
    resolved["schedule_out"].write_text(schedule_payload, encoding="utf-8")
    schedule_sha = _sha(resolved["schedule_out"])

    protocol = {
        "schema_version": "kinofail.realistic-c4-direct-protocol.v1",
        "protocol_id": "kinofail-realistic-c4-direct-final-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_any_direct_c4_action_outcome",
        "estimand": "selective_policy_minus_always_safe_policy",
        "schedule": str(resolved["schedule_out"].relative_to(ROOT)),
        "schedule_sha256": schedule_sha,
        "collector": str(resolved["collector"].relative_to(ROOT)),
        "collector_sha256": _sha(resolved["collector"]),
        "scene_registry": str(resolved["scene_registry"].relative_to(ROOT)),
        "scene_registry_sha256": _sha(resolved["scene_registry"]),
        "shared_actor": str(resolved["actor"].relative_to(ROOT)),
        "shared_actor_sha256": _sha(resolved["actor"]),
        "source_schedule": str(resolved["source_schedule"].relative_to(ROOT)),
        "source_schedule_sha256": _sha(resolved["source_schedule"]),
        "snapshot_records": str(resolved["snapshot_records"].relative_to(ROOT)),
        "snapshot_records_sha256": _sha(resolved["snapshot_records"]),
        "feature_manifest": str(resolved["feature_manifest"].relative_to(ROOT)),
        "feature_manifest_sha256": _sha(resolved["feature_manifest"]),
        "prediction_artifact": str(resolved["predictions"].relative_to(ROOT)),
        "prediction_artifact_sha256": _sha(resolved["predictions"]),
        "model_checkpoints": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha(path),
            }
            for path in checkpoint_paths
        ],
        "case_count": len(schedule),
        "cases_per_scene": len(schedule) // len(test_scenes),
        "physical_episode_count": 2 * len(schedule),
        "scene_clusters": sorted(test_scenes),
        "domains": sorted(test_scenes.values()),
        "operators": sorted(TARGET_OPERATORS),
        "nuisance_realizations_per_scene_operator": 5,
        "frozen_decision_rule": {
            "model": "mean probability from five structured_bidirectional seeds",
            "target": TARGET_ATTRIBUTION,
            "probability_threshold": RELEASE_THRESHOLD,
            "minimum_seed_agreement": REQUIRED_SEED_AGREEMENT,
            "release_action": "backstep_release",
            "fallback_action": "hold_and_request",
            "rule_selected_from_direct_test_outcomes": False,
            "direct_test_outcomes_available_at_freeze": False,
        },
        "freeze_audit": {
            "selection_uses_model_output": False,
            "selection_uses_test_truth": False,
            "all_predeclared_cases_retained": len(schedule) == 75,
            "release_count": release_count,
            "release_coverage": release_count / len(schedule),
            "release_cell_counts": dict(sorted(release_counts.items())),
        },
        "allowed_decision_inputs": ["rgb_window", "proprio_window"],
        "forbidden_decision_inputs": [
            "truth_attribution",
            "operator",
            "scene_cluster",
            "domain",
            "split",
            "severity",
            "nuisance_id",
            "test_outcome",
            "terminal_cost",
        ],
        "bootstrap": {"draws": 20_000, "seed": 2026072405},
        "acceptance": {
            "minimum_paired_cases": 75,
            "minimum_scene_clusters": 3,
            "minimum_domains": 3,
            "minimum_release_coverage": 0.15,
            "minimum_released_attribution_precision": 0.95,
        },
        "outcome_policy": (
            "retain all weak, null, harmful, fallen and failed cases; no threshold, "
            "action, severity, scene or case changes after this freeze"
        ),
        "claim_boundary": (
            "Primary C4 is a supported-action claim on the five predeclared "
            "operators in three held-out realistic scene clusters. It does not "
            "claim arbitrary open-world recovery or universal physical safety."
        ),
    }
    resolved["protocol_out"].parent.mkdir(parents=True, exist_ok=True)
    resolved["protocol_out"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_path = resolved["schedule_out"].with_name("freeze_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "passed": True,
                "protocol": str(resolved["protocol_out"].relative_to(ROOT)),
                "protocol_sha256": _sha(resolved["protocol_out"]),
                "schedule": str(resolved["schedule_out"].relative_to(ROOT)),
                "schedule_sha256": schedule_sha,
                "case_count": len(schedule),
                "release_count": release_count,
                "release_coverage": release_count / len(schedule),
                "selection_used_direct_outcomes": False,
                "direct_outcomes_existed_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
