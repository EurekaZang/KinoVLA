#!/usr/bin/env python3
"""Build the expanded actual-action A4 design before F36 inference.

One model-blind valid Scale group is selected for each
scene x operator x severity cell.  The 300 physical cases each receive seven
matched action arms (continue, always-safe halt, and all five recovery labels),
giving 2,100 actual-action Isaac episodes across all 30 confirmation scenes.
The full label-swap matrix lets a frozen attribution policy be evaluated without
substituting an oracle action when its predicted cause is wrong.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
F35 = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"
SCALE = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scale_schedule.jsonl"
F33 = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/schedule.jsonl"
F25_AUDIT = ROOT / "outputs/kinofail_replenishment_f25/final_audit.json"
F25_SCHEDULE_ROOT = ROOT / "outputs/kinofail_replenishment_f25/schedules"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7"
OPERATORS = (
    "O2_compliance",
    "O4_tether",
    "O5_payload",
    "O8_invisible_collider",
    "O9_high_centering",
)
SEVERITIES = ("moderate", "hard")
RECOVERY = {
    "O2_compliance": "stabilize_slow_cross",
    "O4_tether": "release_detour_replan",
    "O5_payload": "lower_stabilize_slow",
    "O8_invisible_collider": "backstep_detour_replan",
    "O9_high_centering": "backstep_detour_replan",
}
ACTIONS = (
    "continue",
    "always_safe_halt",
    "recover_as_O2_compliance",
    "recover_as_O4_tether",
    "recover_as_O5_payload",
    "recover_as_O8_invisible_collider",
    "recover_as_O9_high_centering",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_f35_primary_decisions() -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Bind each primary anomaly snapshot to the exact 80-D policy input."""

    valid_groups: set[str] = set()
    decision_meta: dict[str, dict[str, Any]] = {}
    for records_path in sorted(
        F35.glob("shards/*/scale/snapshots/snapshot_records.jsonl")
    ):
        feature_path = (
            records_path.parents[1] / "unified_features" / "features.npz"
        )
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        with np.load(feature_path, allow_pickle=False) as archive:
            sample_ids = archive["sample_ids"].astype(str)
            proprio = np.asarray(archive["proprio"], dtype=np.float32)
        features_by_sample = {
            str(sample_id): proprio[index]
            for index, sample_id in enumerate(sample_ids)
        }
        records = jsonl(records_path)
        if (
            len(features_by_sample) != len(records)
            or proprio.shape != (len(records), 80)
            or not np.isfinite(proprio).all()
        ):
            raise RuntimeError(f"invalid F35 proprio shard: {feature_path}")
        for snapshot in records:
            group_id = str(snapshot["counterfactual_group_id"])
            valid_groups.add(group_id)
            if (
                snapshot["condition"] == "anomaly"
                and snapshot["appearance_intervention_id"] == "primary"
            ):
                if group_id in decision_meta:
                    raise RuntimeError(
                        f"duplicate F35 primary anomaly sample: {group_id}"
                    )
                sample_id = str(snapshot["sample_id"])
                expected = features_by_sample[sample_id]
                timestamps = [
                    float(value) for value in snapshot["proprio_timestamp_s"]
                ]
                if len(timestamps) != 21:
                    raise RuntimeError(f"invalid F35 proprio window: {sample_id}")
                decision_meta[group_id] = {
                    "sample_id": sample_id,
                    "decision_time_s": float(snapshot["decision_time_s"]),
                    "event_time_s": float(
                        snapshot[
                            "event_time_s_from_anomaly_privileged_telemetry"
                        ]
                    ),
                    "source_manifest_sha256": str(
                        snapshot["source_manifest_sha256"]
                    ),
                    "proprio_timestamp_s": timestamps,
                    "invariant_proprio_80": expected.astype(float).tolist(),
                    "invariant_proprio_80_sha256": array_sha256(expected),
                    "f35_feature_archive": str(feature_path.relative_to(ROOT)),
                    "f35_feature_archive_sha256": sha256(feature_path),
                }
    return valid_groups, decision_meta


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    f35_audit_path = F35 / "overlay_audit.json"
    if not f35_audit_path.is_file():
        raise FileNotFoundError(f35_audit_path)
    f35_audit = json.loads(f35_audit_path.read_text())
    if (
        f35_audit.get("passed") is not True
        or f35_audit.get("model_prediction_or_score_read") is not False
        or f35_audit.get("legacy_o9_features_excluded") is not True
    ):
        raise RuntimeError("A4 design requires the valid model-blind F35 overlay")

    valid_groups, f35_decision_meta = load_f35_primary_decisions()
    original_anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    f33_by_source = {
        str(row["o9_semantic_recollection"]["source_counterfactual_group_id"]): row
        for row in jsonl(F33)
        if row["condition"] == "anomaly"
    }
    f25_audit = json.loads(F25_AUDIT.read_text())
    f25_replacement_by_original = {
        str(row["original_counterfactual_group_id"]): str(
            row["replacement_counterfactual_group_id"]
        )
        for row in f25_audit["accepted"]
        if str(row["target_operator"]) != "O9_high_centering"
    }
    f25_schedule_rows = [
        row
        for path in sorted(F25_SCHEDULE_ROOT.glob("*/scale/schedule.jsonl"))
        for row in jsonl(path)
    ]
    f25_anomaly_by_replacement = {
        str(row["counterfactual_group_id"]): row
        for row in f25_schedule_rows
        if row["condition"] == "anomaly"
    }
    if len(original_anomaly) != 10_560 or len(f33_by_source) != 960:
        raise RuntimeError("A4 source schedules are incomplete")

    candidates: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id in valid_groups:
        row = original_anomaly[group_id]
        key = (str(row["scene_id"]), str(row["target_operator"]), str(row["severity_id"]))
        if key[1] in OPERATORS and key[2] in SEVERITIES:
            candidates[key].append(group_id)

    scenes = sorted(f35_audit.get("pairs_by_scene", {}))
    if len(scenes) != 30:
        raise RuntimeError("A4 requires all 30 F35 scenes")
    cases: list[dict[str, Any]] = []
    for scene in scenes:
        for operator in OPERATORS:
            for severity in SEVERITIES:
                options = candidates[(scene, operator, severity)]
                if not options:
                    raise RuntimeError(f"no valid A4 source: {scene}/{operator}/{severity}")
                # Hash order is fixed before inference and is independent of outcomes.
                group_id = min(
                    options,
                    key=lambda value: stable_key("a4-v7", scene, operator, severity, value),
                )
                original = original_anomaly[group_id]
                replacement_id = f25_replacement_by_original.get(group_id)
                if operator == "O9_high_centering":
                    source = f33_by_source[group_id]
                elif replacement_id is not None:
                    source = f25_anomaly_by_replacement[replacement_id]
                else:
                    source = original
                case_id = f"a4v7__{scene}__{operator}__{severity}"
                nuisance = source.get("physical_nuisance", {})
                reset_seed = int(nuisance.get("physics_seed", source["operator_seed"]))
                if reset_seed != int(source["operator_seed"]):
                    raise RuntimeError(f"source nuisance seed drift: {group_id}")
                cases.append(
                    {
                        "schema_version": "kinofail.reconfirmation-a4-v7-schedule.v1",
                        "case_id": case_id,
                        "scene_id": scene,
                        "domain": str(original["domain"]),
                        "operator": operator,
                        "severity_id": severity,
                        "source_counterfactual_group_id": group_id,
                        "source_slot_episode_id": str(original["episode_id"]),
                        "source_physical_episode_id": str(source["episode_id"]),
                        "source_record": source,
                        "source_is_f33_direct_o9": operator == "O9_high_centering",
                        "source_is_f25_replacement": replacement_id is not None,
                        "reset_seed": reset_seed,
                        "source_physical_nuisance": nuisance,
                        "source_f35_decision": f35_decision_meta[group_id],
                        "actions": list(ACTIONS),
                        "operator_recovery": RECOVERY[operator],
                        "pairing": "same scene/operator/parameters/reset seed, source-matched predecision controller, and certified F35 policy observation",
                        "source_observation_replay_required": True,
                        "selection_used_model_predictions_or_outcomes": False,
                    }
                )
    if len(cases) != 300:
        raise AssertionError(len(cases))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases))
    audit = {
        "schema_version": "kinofail.reconfirmation-a4-v7-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_model_blind_design_before_f36_prediction",
        "passed": True,
        "model_prediction_or_outcome_read": False,
        "selection_used_model_predictions_or_outcomes": False,
        "counts": {
            "scenes": len(scenes),
            "domains": len({row["domain"] for row in cases}),
            "operators": len({row["operator"] for row in cases}),
            "severity_strata": len({row["severity_id"] for row in cases}),
            "physical_cases": len(cases),
            "action_arms": len(ACTIONS),
            "physical_episodes": len(cases) * len(ACTIONS),
        },
        "actions": {
            "common": list(ACTIONS),
            "operator_recovery": RECOVERY,
        },
        "source_observation_replay": {
            "required": True,
            "window": "same 21x19 raw proprio timestamps as the F35 anomaly-primary sample",
            "comparison": "recomputed 80-D invariant summary against the exact F35 policy input",
            "absolute_tolerance": 0.005,
            "relative_tolerance": 0.005,
            "failure_policy": "fail the physical case; no outcome-dependent recollection",
        },
        "local_recovery_endpoint": {
            "progress_beyond_frozen_decision_state_m": 0.75,
            "maximum_absolute_route_lateral_offset_m": 0.25,
            "terminates_on_first_success": True,
            "downstream_scene_outcomes_after_success_are_out_of_scope": True,
        },
        "source_sha256": {
            "f35_overlay_audit": sha256(f35_audit_path),
            "scale_schedule": sha256(SCALE),
            "f33_schedule": sha256(F33),
            "f25_final_audit": sha256(F25_AUDIT),
            "f25_schedules": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(F25_SCHEDULE_ROOT.glob("*/scale/schedule.jsonl"))
            },
            "builder": sha256(Path(__file__).resolve()),
        },
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
    }
    (OUTPUT / "design_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
