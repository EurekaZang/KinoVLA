#!/usr/bin/env python3
"""Build model-blind pilot/formal schedules for the full action matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.o9_semantics import evaluate_o9_high_centering
from kino_vla.eval.c2_temporal_v5 import invariant_summary


RAW_ROOT = Path("/data/eureka/kinofail_kino_v4_confirmation_v1")
RECORDS = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d/scale/records.jsonl"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_action_full_v1.py"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"

OPERATORS = (
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
)
ACTIONS = (
    "continue",
    "always_safe_halt",
    "recover_as_low_friction",
    "recover_as_O2_compliance",
    "recover_as_O3_collapse",
    "recover_as_O4_tether",
    "recover_as_hold_request",
    "recover_as_O6_push",
    "recover_as_O8_invisible_collider",
    "recover_as_O9_high_centering",
    "recover_as_O10_effort_decay",
)
CORRECT_ACTION = {
    "O1_mu_field": "recover_as_low_friction",
    "O2_compliance": "recover_as_O2_compliance",
    "O3_collapse": "recover_as_O3_collapse",
    "O4_tether": "recover_as_O4_tether",
    "O5_payload": "recover_as_hold_request",
    "O6_push": "recover_as_O6_push",
    "O7_visual_remap": "recover_as_low_friction",
    "O8_invisible_collider": "recover_as_O8_invisible_collider",
    "O9_high_centering": "recover_as_O9_high_centering",
    "O10_effort_decay": "recover_as_O10_effort_decay",
    "O11_obs_bias": "recover_as_hold_request",
}
DECISION_DWELL_STEPS = {
    "O1_mu_field": 8,
    "O2_compliance": 1,
    "O3_collapse": 1,
    "O4_tether": 3,
    "O5_payload": 1,
    "O6_push": 1,
    "O7_visual_remap": 8,
    "O8_invisible_collider": 1,
    "O9_high_centering": 1,
    "O10_effort_decay": 1,
    "O11_obs_bias": 20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def source_dir(record: dict[str, Any]) -> Path:
    return (
        RAW_ROOT
        / "corpus"
        / str(record["scene_cluster"])
        / "scale"
        / str(record["material_family"])
        / str(record["target_operator"])
        / str(record["physical_episode_id"])
    )


def action_boundary_event(
    operator_name: str, privileged_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the first pre-registered, physically recoverable event row."""

    for row in privileged_rows:
        telemetry = dict(row.get("operator", {}))
        time_s = float(row.get("timestamp_s", 0.0))
        passed = False
        certificate: dict[str, Any] | None = None
        if operator_name in {"O1_mu_field", "O7_visual_remap"}:
            passed = int(telemetry.get("scale_region_exposure_steps", 0)) >= 8
            if operator_name == "O7_visual_remap":
                passed = passed and int(telemetry.get("depth_fault_count", 0)) > 0
        elif operator_name == "O2_compliance":
            passed = max(
                (int(foot.get("contact_steps", 0)) for foot in telemetry.get("feet", [])),
                default=0,
            ) >= 1
        elif operator_name == "O3_collapse":
            passed = any(
                region.get("collapsed") is True
                for region in telemetry.get("regions", [])
            )
        elif operator_name == "O4_tether":
            passed = bool(
                int(telemetry.get("total_attachment_cycles", 0)) > 0
                and int(telemetry.get("active_feet", 0)) > 0
                and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
            )
        elif operator_name == "O5_payload":
            passed = bool(
                time_s >= 0.02 and float(telemetry.get("payload_kg", 0.0)) > 0.0
            )
        elif operator_name == "O6_push":
            passed = bool(
                telemetry.get("complete") is True
                and int(telemetry.get("applied_steps", 0)) > 0
            )
        elif operator_name == "O8_invisible_collider":
            passed = bool(
                int(telemetry.get("scale_region_exposure_steps", 0)) >= 1
                and any(
                    obstacle.get("collision_requested") is True
                    for obstacle in telemetry.get("obstacles", [])
                )
            )
        elif operator_name == "O9_high_centering":
            certificate = evaluate_o9_high_centering(telemetry)
            passed = bool(certificate["passed"])
        elif operator_name == "O10_effort_decay":
            passed = float(telemetry.get("effort_scale", 1.0)) < 0.95
        elif operator_name == "O11_obs_bias":
            passed = bool(
                time_s >= 0.40
                and telemetry.get("raw_pipeline_active") is True
                and telemetry.get("initialized") is True
            )
        else:
            raise ValueError(operator_name)
        if passed:
            return row, certificate
    return None, None


def source_bundle(
    record: dict[str, Any], *, relaxed_o4_runtime_admission: bool = False
) -> dict[str, Any]:
    episode = source_dir(record)
    manifest_path = episode / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256(manifest_path) != str(record["source_manifest_sha256"]):
        raise RuntimeError(f"source manifest hash mismatch: {manifest_path}")
    manifest = json_object(manifest_path)
    operator_name = str(record["target_operator"])
    telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
    o4_engaged_source = bool(
        operator_name == "O4_tether"
        and manifest.get("runtime_validation", {}).get("issues")
        == ["operator_local_qa_failed"]
        and int(telemetry.get("total_attachment_cycles", 0)) > 0
        and int(telemetry.get("active_feet", 0)) > 0
        and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
    )
    ordinary_admission = bool(
        manifest.get("runtime_validation", {}).get("passed") is True
        and manifest.get("runtime_validation", {}).get("evaluation_eligible") is True
        and manifest.get("operator_readback", {}).get("qa_passed") is True
    )
    relaxed_o4_admission = bool(
        relaxed_o4_runtime_admission
        and operator_name == "O4_tether"
        and manifest.get("runtime_validation", {}).get("issues")
        == ["operator_local_qa_failed"]
    )
    if (
        not (ordinary_admission or o4_engaged_source or relaxed_o4_admission)
        or manifest.get("geometry_readback", {}).get("qa_passed") is not True
    ):
        raise RuntimeError(f"source failed runtime admission: {manifest_path}")

    with np.load(episode / "proprio.npz", allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamp_s"], dtype=np.float64)
    target_times = np.asarray(record["proprio_timestamp_s"], dtype=np.float64)
    selected = np.asarray(
        [int(np.argmin(np.abs(timestamps - target))) for target in target_times],
        dtype=np.int64,
    )
    if (
        len(set(selected.tolist())) != 21
        or float(np.max(np.abs(timestamps[selected] - target_times))) > 1.0e-6
    ):
        raise RuntimeError(f"source proprio timestamps do not replay: {episode}")
    window = np.asarray(features[selected], dtype=np.float32)
    summary = invariant_summary(window)
    if window.shape != (21, 19) or summary.shape != (80,):
        raise RuntimeError(f"source observation shape mismatch: {episode}")

    telemetry = manifest["operator_readback"]["telemetry"]
    strict_o9 = (
        evaluate_o9_high_centering(telemetry)
        if record["target_operator"] == "O9_high_centering"
        else None
    )
    nuisance = dict(manifest["collection"]["physical_nuisance"])
    privileged_rows = jsonl(episode / "privileged.jsonl")
    event_row, event_certificate = action_boundary_event(
        str(record["target_operator"]), privileged_rows
    )
    operator_seed = int(nuisance["physics_seed"])
    source_record = {
        "target_operator": str(record["target_operator"]),
        "physics_parameters": dict(
            manifest["operator_readback"]["applied_parameters"]
        ),
        "condition": "anomaly",
        "physical_realization": str(
            manifest["geometry_readback"]["physical_realization"]
        ),
        "physical_nuisance": nuisance,
        "operator_seed": operator_seed,
    }
    return {
        "episode": episode,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "source_record": source_record,
        "strict_o9": strict_o9,
        "action_event_row": event_row,
        "action_event_certificate": event_certificate,
        "source_decision": {
            "sample_id": f"{record['physical_episode_id']}__action_boundary",
            "decision_time_s": (
                None if event_row is None else float(event_row["timestamp_s"])
            ),
            "event_time_s": (
                None if event_row is None else float(event_row["timestamp_s"])
            ),
            "certificate_mode": "same_run_action_boundary",
            "external_snapshot_replay_claimed": False,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path),
        },
        "decision_state": {
            "last_base_height_m": (
                float("nan")
                if event_row is None
                else float(event_row["base_height_m"])
            ),
            "last_absolute_tilt_rad": (
                float("nan")
                if event_row is None
                else abs(float(event_row["tilt_rad"]))
            ),
            "source_window_shape": None,
            "action_boundary_event_present": event_row is not None,
            "source_admission": (
                "o4_relaxed_dynamic_attachment_precursor"
                if relaxed_o4_admission and not o4_engaged_source
                else "o4_engaged_pre_release_state"
                if o4_engaged_source
                else "runtime_validation_passed"
            ),
        },
    }


def eligible_candidates(
    *,
    relaxed_o9_action_precursor: bool = False,
    relaxed_o4_action_precursor: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [
        row
        for row in jsonl(RECORDS)
        if row.get("condition") == "anomaly"
        and row.get("appearance_intervention_id") == "primary"
        and row.get("target_operator") in OPERATORS
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in primary:
        try:
            bundle = source_bundle(
                record,
                relaxed_o4_runtime_admission=relaxed_o4_action_precursor,
            )
        except RuntimeError as error:
            if not str(error).startswith("source failed runtime admission:"):
                raise
            rejected.append(
                {
                    "sample_id": record["sample_id"],
                    "scene_id": record["scene_cluster"],
                    "operator": record["target_operator"],
                    "severity_id": record["severity_id"],
                    "reasons": ["source_runtime_admission_failed"],
                }
            )
            continue
        state = bundle["decision_state"]
        reasons: list[str] = []
        relaxed_precursor = bool(
            (
                relaxed_o9_action_precursor
                and record["target_operator"] == "O9_high_centering"
            )
            or (
                relaxed_o4_action_precursor
                and record["target_operator"] == "O4_tether"
            )
        )
        if bundle["action_event_row"] is None and not relaxed_precursor:
            reasons.append("registered_action_boundary_event_missing")
        if state["last_base_height_m"] < 0.23:
            reasons.append("source_base_height_below_0.23_m")
        if state["last_absolute_tilt_rad"] >= 0.55:
            reasons.append("source_absolute_tilt_at_least_0.55_rad")
        if (
            record["target_operator"] == "O9_high_centering"
            and not relaxed_o9_action_precursor
            and not bool(bundle["strict_o9"]["passed"])
        ):
            reasons.append("strict_high_centering_semantics_not_proven")
        enriched = {**record, "_bundle": bundle}
        if reasons:
            rejected.append(
                {
                    "sample_id": record["sample_id"],
                    "scene_id": record["scene_cluster"],
                    "operator": record["target_operator"],
                    "severity_id": record["severity_id"],
                    "reasons": reasons,
                }
            )
        else:
            accepted.append(enriched)
    return accepted, rejected


def select_pilot(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for index, operator in enumerate(OPERATORS):
        rows = [row for row in candidates if row["target_operator"] == operator]
        if not rows:
            raise RuntimeError(f"no admissible pilot source for {operator}")
        preferred_domain = ("life", "production", "wild")[index % 3]
        preferred = [row for row in rows if row["domain"] == preferred_domain]
        pool = preferred or rows
        selected.append(
            min(
                pool,
                key=lambda row: stable_key(
                    "action-full-v1-pilot",
                    operator,
                    row["scene_cluster"],
                    row["counterfactual_group_id"],
                ),
            )
        )
    return selected


def select_formal(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    scenes = sorted({str(row["scene_cluster"]) for row in candidates})
    missing = []
    for scene in scenes:
        for operator in OPERATORS:
            for severity in ("moderate", "hard"):
                rows = [
                    row
                    for row in candidates
                    if row["scene_cluster"] == scene
                    and row["target_operator"] == operator
                    and row["severity_id"] == severity
                ]
                if not rows:
                    missing.append((scene, operator, severity))
                    continue
                selected.append(
                    min(
                        rows,
                        key=lambda row: stable_key(
                            "action-full-v1-formal",
                            scene,
                            operator,
                            severity,
                            row["counterfactual_group_id"],
                        ),
                    )
                )
    if missing:
        preview = ", ".join("/".join(row) for row in missing[:20])
        raise RuntimeError(
            f"formal coverage incomplete ({len(missing)} cells): {preview}"
        )
    if len(scenes) != 12 or len(selected) != 12 * 11 * 2:
        raise RuntimeError(
            f"formal design must contain 264 cases over 12 scenes, got "
            f"{len(selected)} over {len(scenes)}"
        )
    return selected


def build_case(record: dict[str, Any], mode: str) -> dict[str, Any]:
    bundle = record["_bundle"]
    operator = str(record["target_operator"])
    return {
        "schema_version": "kinofail.action-full-v1-schedule.v1",
        "case_id": (
            f"actionv1__{mode}__{record['scene_cluster']}__{operator}__"
            f"{record['severity_id']}__{record['counterfactual_group_id']}"
        ),
        "scene_id": str(record["scene_cluster"]),
        "domain": str(record["domain"]),
        "operator": operator,
        "severity_id": str(record["severity_id"]),
        "source_counterfactual_group_id": str(record["counterfactual_group_id"]),
        "source_slot_episode_id": str(record["physical_episode_id"]),
        "source_physical_episode_id": str(record["physical_episode_id"]),
        "source_record": bundle["source_record"],
        "source_is_f33_direct_o9": False,
        "reset_seed": int(
            bundle["source_record"]["physical_nuisance"]["physics_seed"]
        ),
        "source_physical_nuisance": bundle["source_record"]["physical_nuisance"],
        "source_f35_decision": bundle["source_decision"],
        "actions": list(ACTIONS),
        "operator_recovery": CORRECT_ACTION[operator],
        "decision_mode": "first_recoverable_operator_event",
        "operator_engagement_dwell_steps": DECISION_DWELL_STEPS[operator],
        "pairing": "one exact physical checkpoint restored across eleven action arms",
        "source_observation_replay_required": False,
        "selection_used_model_predictions_or_action_outcomes": False,
        "decision_state_eligibility": bundle["decision_state"],
        "strict_o9_semantic_certificate": bundle["strict_o9"],
        "action_boundary_semantic_certificate": bundle[
            "action_event_certificate"
        ],
        "development_only": mode == "pilot",
        "counts_as_publication_evidence": mode == "formal",
    }


def scene_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def locate(scene: str, filename: str, expected_sha256: str) -> Path:
        candidates = [
            path
            for path in (RAW_ROOT / "scenes").rglob(filename)
            if scene in path.parts and sha256(path) == expected_sha256
        ]
        if not candidates:
            raise RuntimeError(
                f"no full-scene hash match for {scene}/{filename}/"
                f"{expected_sha256}"
            )
        return min(candidates, key=lambda path: (len(path.parts), str(path)))

    def usd_closure(root_usd: Path) -> list[Path]:
        pending = [root_usd]
        visited: set[Path] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            if not current.is_file():
                raise RuntimeError(f"unresolved USD dependency: {current}")
            visited.add(current)
            text = current.read_text(encoding="utf-8", errors="ignore")
            for value in re.findall(r"@([^@]+)@", text):
                value_path = Path(value)
                if value_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                    continue
                dependency = (
                    value_path if value_path.is_absolute() else current.parent / value_path
                ).resolve()
                if dependency not in visited:
                    pending.append(dependency)
        return sorted(visited)

    registry = []
    for scene in sorted({str(row["scene_cluster"]) for row in rows}):
        exemplars = [row for row in rows if row["scene_cluster"] == scene]
        artifacts = exemplars[0]["_bundle"]["manifest"]["scene_readback"]["artifacts"]
        by_name = {Path(row["path"]).name: row for row in artifacts}
        usd_artifact = by_name.get("episode_terrain_v2.usda") or by_name.get(
            "episode.usda"
        )
        if usd_artifact is None:
            raise RuntimeError(f"scene provenance has no episode USD: {scene}")
        usd = locate(
            scene,
            Path(usd_artifact["path"]).name,
            str(usd_artifact["sha256"]),
        )
        audit = locate(
            scene,
            "compiled_scene_audit.json",
            str(by_name["compiled_scene_audit.json"]["sha256"]),
        )
        if (
            sha256(usd) != usd_artifact["sha256"]
            or sha256(audit) != by_name["compiled_scene_audit.json"]["sha256"]
        ):
            raise RuntimeError(f"scene provenance hash mismatch: {scene}")
        required_layers = usd_closure(usd)
        registry.append(
            {
                "scene_id": scene,
                "episode_usd": str(usd),
                "episode_sha256": sha256(usd),
                "compiled_audit": str(audit),
                "compiled_audit_sha256": sha256(audit),
                "domain": str(exemplars[0]["domain"]),
                "usd_dependency_closure": [str(path) for path in required_layers],
            }
        )
    return {
        "schema_version": "kinofail.action-full-v1-scene-registry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scenes": registry,
    }


def asset_record(path: str, **extra: Any) -> dict[str, Any]:
    resolved = ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": path, "sha256": sha256(resolved), **extra}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / f"outputs/kinofail_action_full_v1_{args.mode}"
    ).resolve()
    if out.exists():
        raise FileExistsError(out)
    candidates, rejected = eligible_candidates()
    selected = (
        select_pilot(candidates)
        if args.mode == "pilot"
        else select_formal(candidates)
    )
    cases = [build_case(row, args.mode) for row in selected]
    registry = scene_registry(selected)
    out.mkdir(parents=True)
    schedule_path = out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    registry_path = out / "scene_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predecision = asset_record(
        "outputs/locomotion/policy.pt",
        purpose="replay_current_v4_predecision_state",
    )
    actor = {
        **asset_record("outputs/locomotion/recovery_route_v1/policy.pt"),
        "training_manifest": "outputs/locomotion/recovery_route_v1/training_manifest.json",
        "training_manifest_sha256": sha256(
            ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
        ),
        "training_freeze": "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json",
        "training_freeze_sha256": sha256(
            ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
        ),
        "sim_config": "configs/sim/go2_skeleton.yaml",
        "sim_config_sha256": sha256(ROOT / "configs/sim/go2_skeleton.yaml"),
        "shared_across_all_action_arms": True,
        "receives_attribution_input": False,
    }
    actor["policy"] = actor.pop("path")
    actor["policy_sha256"] = actor.pop("sha256")
    predecision["policy"] = predecision.pop("path")
    predecision["policy_sha256"] = predecision.pop("sha256")
    protocol = {
        "schema_version": "kinofail.action-full-v1-protocol.v1",
        "protocol_id": f"kinofail-action-full-v1-{args.mode}-20260809",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_pilot" if args.mode == "pilot" else "formal_frozen",
        "development_only": args.mode == "pilot",
        "schedule": str(schedule_path),
        "schedule_sha256": sha256(schedule_path),
        "scene_registry": str(registry_path),
        "scene_registry_sha256": sha256(registry_path),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "base_collector": str(BASE_COLLECTOR.relative_to(ROOT)),
        "base_collector_sha256": sha256(BASE_COLLECTOR),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "predecision_actor": predecision,
        "low_level_actor": actor,
        "counts": {
            "scenes": len(registry["scenes"]),
            "operators": len(OPERATORS),
            "physical_cases": len(cases),
            "action_arms": len(ACTIONS),
            "recovery_action_arms": len(ACTIONS) - 2,
            "physical_episodes": len(cases) * len(ACTIONS),
        },
        "operators": list(OPERATORS),
        "action_arms": list(ACTIONS),
        "operator_to_registered_action": CORRECT_ACTION,
        "branching_contract": {
            "one_shared_prefix_per_case": True,
            "all_eleven_arms_restore_one_checkpoint": True,
            "same_low_level_actor": True,
            "complete_outcome_retention": True,
            "no_result_dependent_retry": True,
        },
        "source_contract": {
            "dataset": "KiNO-Fail v4 Scale",
            "raw_root": str(RAW_ROOT),
            "record_path": str(RECORDS.relative_to(ROOT)),
            "record_sha256": sha256(RECORDS),
            "selection_uses_model_predictions": False,
            "selection_uses_action_outcomes": False,
            "strict_o9_semantic_gate": True,
        },
    }
    protocol_path = out / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in rejected:
        for reason in row["reasons"]:
            rejection_counts[reason] += 1
    audit = {
        "schema_version": "kinofail.action-full-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "mode": args.mode,
        "counts": protocol["counts"],
        "selected_case_ids": [row["case_id"] for row in cases],
        "selection_used_model_predictions_or_action_outcomes": False,
        "source_candidates_admitted": len(candidates),
        "source_candidates_rejected": len(rejected),
        "source_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "strict_o9_selected": sum(
            row["operator"] == "O9_high_centering"
            and row["strict_o9_semantic_certificate"]["passed"] is True
            for row in cases
        ),
        "hashes": {
            "schedule": sha256(schedule_path),
            "scene_registry": sha256(registry_path),
            "protocol": sha256(protocol_path),
            "collector": sha256(COLLECTOR),
            "base_collector": sha256(BASE_COLLECTOR),
        },
    }
    (out / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
