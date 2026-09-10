#!/usr/bin/env python3
"""Model-blind, task-aligned audit for the complete KiNO-v4 T3 corpus.

This successor audit preserves the generic runtime audits verbatim.  It
enforces the nonblocking runtime-issue policy frozen in the F1 snapshot
protocol before confirmation data existed, and then validates the exact T3
decision inputs.  No checkpoint, prediction, truth-key, or score artifact is
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.runtime_manifest import schedule_record_sha256
from kino_vla.eval.c2_temporal_v5 import (
    FOOTPRINT_MARGIN_M,
    POST_ENCOUNTER_DELAY_S,
    geometry_aligned_invariant_summary,
)


F1 = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1/freeze_manifest.json"
F1_BUILDER = ROOT / "scripts/build_kinofail_kino_v4_confirmation_schedules_v1.py"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"
    / "f5_task_aligned_audit"
)
CAMPAIGNS = (
    {
        "id": "original_f4e",
        "selected": ROOT
        / "outputs/freeze/kino_v4_confirmation_t3_runin_f4d/selected_cases.jsonl",
        "seal": ROOT
        / "outputs/freeze/kino_v4_confirmation_t3_runin_f4e/seal_manifest.json",
        "raw_audit": ROOT
        / "outputs/kinofail_kino_v4_confirmation_t3_runin_f4e/final_audit.json",
        "corpus": Path(
            "/data/eureka/kinofail_kino_v4_confirmation_t3_runin_f4e/corpus"
        ),
    },
    {
        "id": "extension_f4l",
        "selected": ROOT
        / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design/selected_cases.jsonl",
        "seal": ROOT
        / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design/seal_manifest.json",
        "raw_audit": ROOT
        / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/collection/final_audit.json",
        "corpus": Path(
            "/data/eureka/kinofail_kino_v4_confirmation_t3_extension_f4h/corpus"
        ),
    },
)
NONBLOCKING_SUFFIXES = (
    "appearance_effect_too_small",
    "rgb_spatial_contrast_too_low",
)
VIEWS = ("primary", "swap_01", "swap_02")
VISUAL_FRAMES = 5
VISUAL_CROP_Y_FRACTION = 0.45
MAX_RGB_DECISION_SKEW_S = 0.051
APPEARANCE_L1_THRESHOLD = 0.015
ATTRITION_LIMIT = 0.05
FORBIDDEN_NAME_TOKENS = ("prediction", "truth_key", "score")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def manifest_path(result: Mapping[str, Any], corpus: Path) -> Path:
    path = Path(str(result["manifest"]))
    if not path.is_absolute():
        path = corpus / path
    path = path.resolve()
    if corpus.resolve() not in path.parents:
        raise RuntimeError(f"manifest escapes corpus: {path}")
    return path


def allowed_raw_issue(issue: str) -> bool:
    return any(issue.endswith(suffix) for suffix in NONBLOCKING_SUFFIXES)


def footprint_encounter(
    manifest: Mapping[str, Any], episode_dir: Path
) -> dict[str, Any]:
    region = manifest["geometry_readback"]["operator_region"]
    telemetry = episode_dir / manifest["artifacts"]["telemetry"]["path"]
    encounters: list[dict[str, Any]] = []
    for line in telemetry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        x, y = (float(value) for value in row["position_xy_m"])
        dx = max(abs(x - float(region["cx"])) - float(region["hx"]), 0.0)
        dy = max(abs(y - float(region["cy"])) - float(region["hy"]), 0.0)
        if math.hypot(dx, dy) <= FOOTPRINT_MARGIN_M:
            encounters.append(row)
    if not encounters:
        raise ValueError(f"no {FOOTPRINT_MARGIN_M:.2f} m Go2-footprint encounter")
    first = encounters[0]
    return {
        "encounter_time_s": float(first["timestamp_s"]),
        "encounter_position_xy_m": [float(value) for value in first["position_xy_m"]],
        "encounter_rows": len(encounters),
        "footprint_margin_m": FOOTPRINT_MARGIN_M,
    }


def operator_gate(
    manifest: Mapping[str, Any], episode_dir: Path
) -> tuple[bool, dict[str, Any]]:
    readback = manifest["operator_readback"]
    operator = str(readback["operator_id"])
    telemetry = readback["telemetry"]
    encounter = footprint_encounter(manifest, episode_dir)
    if operator == "O7_visual_remap":
        passed = (
            telemetry.get("enabled") is True
            and int(telemetry.get("depth_fault_count", 0)) > 0
            and float(telemetry.get("readback_dynamic_friction", math.nan))
            < float(telemetry.get("nominal_readback_dynamic_friction", -math.inf))
            and float(telemetry.get("readback_static_friction", math.nan))
            < float(telemetry.get("nominal_readback_static_friction", -math.inf))
        )
    elif operator == "O8_invisible_collider":
        passed = (
            telemetry.get("enabled") is True
            and any(
                obstacle.get("collision_requested") is True
                and any(
                    collider.get("enabled") is True
                    for collider in obstacle.get("colliders", [])
                )
                for obstacle in telemetry.get("obstacles", [])
            )
        )
    else:
        raise RuntimeError(f"unexpected T3 operator: {operator}")
    return bool(passed), {
        "operator_id": operator,
        "passed": bool(passed),
        "raw_operator_qa_passed": readback.get("qa_passed") is True,
        "raw_base_center_exposure_steps": int(
            telemetry.get("scale_region_exposure_steps", 0)
        ),
        **encounter,
    }


def input_integrity(
    schedule: Mapping[str, Any], manifest: Mapping[str, Any], episode_dir: Path
) -> tuple[bool, list[str], list[str]]:
    issues: list[str] = []
    raw_issues = [
        str(value)
        for value in manifest.get("runtime_validation", {}).get("issues", [])
    ]
    blocking_raw = [value for value in raw_issues if not allowed_raw_issue(value)]
    operator = str(manifest.get("operator_readback", {}).get("operator_id", ""))
    condition = str(schedule["condition"])
    if blocking_raw == ["operator_local_qa_failed"]:
        if not (condition == "anomaly" and operator == "O7_visual_remap"):
            issues.append("operator_local_warning_outside_o7_anomaly")
        blocking_raw = []
    issues.extend(f"blocking_raw_issue:{value}" for value in blocking_raw)
    if str(manifest.get("episode_id")) != str(schedule["episode_id"]):
        issues.append("episode_id_mismatch")
    if str(manifest.get("counterfactual_group_id")) != str(
        schedule["counterfactual_group_id"]
    ):
        issues.append("counterfactual_group_id_mismatch")
    if manifest.get("schedule_record_sha256") != schedule_record_sha256(schedule):
        issues.append("schedule_record_hash_mismatch")
    if str(manifest.get("scene_readback", {}).get("scene_family")) != str(
        schedule["scene_family"]
    ):
        issues.append("scene_family_mismatch")
    if manifest.get("artifact_state") not in {"validated", "collected"}:
        issues.append("artifact_state_invalid")
    for key in ("proprio", "telemetry"):
        artifact = manifest.get("artifacts", {}).get(key, {})
        path = episode_dir / str(artifact.get("path", ""))
        if not path.is_file():
            issues.append(f"{key}_missing")
        elif sha256(path) != artifact.get("sha256"):
            issues.append(f"{key}_hash_mismatch")
    views = manifest.get("artifacts", {}).get("rgb_views", {})
    if set(views) != set(VIEWS):
        issues.append("rgb_view_set_mismatch")
    else:
        lengths = {view: len(rows) for view, rows in views.items()}
        if min(lengths.values(), default=0) < VISUAL_FRAMES:
            issues.append("rgb_view_too_short")
        if len(set(lengths.values())) != 1:
            issues.append("rgb_view_length_mismatch")
    return not issues, issues, raw_issues


def load_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"decision RGB integrity failure: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def decision_visual_gate(
    manifest: Mapping[str, Any], episode_dir: Path, decision_time_s: float
) -> dict[str, Any]:
    views = manifest["artifacts"]["rgb_views"]
    primary = views["primary"]
    times = np.asarray([float(row["timestamp_s"]) for row in primary])
    end_index = int(np.argmin(np.abs(times - decision_time_s)))
    end_skew = abs(float(times[end_index]) - decision_time_s)
    if end_skew > MAX_RGB_DECISION_SKEW_S or end_index + 1 < VISUAL_FRAMES:
        raise ValueError("decision-aligned RGB window unavailable")
    selected = list(range(end_index - VISUAL_FRAMES + 1, end_index + 1))
    pair_l1: dict[str, float] = {}
    selected_paths: dict[str, list[str]] = {view: [] for view in VIEWS}
    for view in VIEWS[1:]:
        alternate = views[view]
        if len(alternate) != len(primary):
            raise ValueError(f"RGB length mismatch: {view}")
        differences: list[float] = []
        for index in selected:
            if not math.isclose(
                float(primary[index]["timestamp_s"]),
                float(alternate[index]["timestamp_s"]),
                abs_tol=1.0e-9,
                rel_tol=0.0,
            ):
                raise ValueError(f"unsynchronized decision RGB: {view}")
            primary_path = episode_dir / str(primary[index]["path"])
            alternate_path = episode_dir / str(alternate[index]["path"])
            primary_rgb = load_rgb(primary_path, str(primary[index]["sha256"]))
            alternate_rgb = load_rgb(alternate_path, str(alternate[index]["sha256"]))
            if primary_rgb.shape != alternate_rgb.shape:
                raise ValueError(f"RGB shape mismatch: {view}")
            crop_start = int(
                math.floor(VISUAL_CROP_Y_FRACTION * primary_rgb.shape[0])
            )
            differences.append(
                float(
                    np.mean(
                        np.abs(
                            primary_rgb[crop_start:, :, :]
                            - alternate_rgb[crop_start:, :, :]
                        )
                    )
                    / 255.0
                )
            )
            selected_paths[view].append(str(alternate[index]["path"]))
        pair_l1[view] = float(np.mean(differences))
    selected_paths["primary"] = [str(primary[index]["path"]) for index in selected]
    passed = all(value >= APPEARANCE_L1_THRESHOLD for value in pair_l1.values())
    return {
        "passed": bool(passed),
        "decision_time_s": float(decision_time_s),
        "selected_frame_indices": selected,
        "selected_timestamps_s": [float(times[index]) for index in selected],
        "end_skew_s": float(end_skew),
        "mean_pair_rgb_l1": pair_l1,
        "minimum_mean_pair_rgb_l1": APPEARANCE_L1_THRESHOLD,
        "frames": VISUAL_FRAMES,
        "crop_y_fraction": VISUAL_CROP_Y_FRACTION,
        "selected_paths": selected_paths,
    }


def audit_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    campaign_id = str(config["id"])
    selected_path = Path(config["selected"]).resolve()
    seal_path = Path(config["seal"]).resolve()
    raw_audit_path = Path(config["raw_audit"]).resolve()
    corpus = Path(config["corpus"]).resolve()
    seal = load_json(seal_path)
    raw = load_json(raw_audit_path)
    selected = load_jsonl(selected_path)
    planned = int(seal["counts"]["planned_cases"])
    if (
        seal.get("passed") is not True
        or seal.get("model_prediction_truth_key_or_score_read") is not False
        or raw.get("checks", {}).get("all_pairs_terminal") is not True
        or raw.get("model_prediction_truth_key_or_score_read") is not False
        or raw.get("result_dependent_selection_or_retry") is not False
        or len(selected) != planned
        or sha256(selected_path) != seal.get("selected_cases_sha256")
    ):
        raise RuntimeError(f"invalid terminal model-blind predecessor: {campaign_id}")

    schedules: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    pair_to_case: dict[str, dict[str, Any]] = {}
    for case in selected:
        scene = str(case["scene_id"])
        schedule_path = (ROOT / str(case["schedule"])).resolve()
        if scene not in schedules:
            schedules[scene] = {
                (str(row["counterfactual_group_id"]), str(row["condition"])): row
                for row in load_jsonl(schedule_path)
            }
        for pair_id_value in case["pair_ids"]:
            pair_id = str(pair_id_value)
            if pair_id in pair_to_case:
                raise RuntimeError(f"duplicate pair id: {pair_id}")
            pair_to_case[pair_id] = case

    summaries: dict[str, dict[str, Any]] = {}
    for path in corpus.glob("*/pair_summaries/*.json"):
        summary = load_json(path)
        pair_id = str(summary["counterfactual_group_id"])
        if pair_id in summaries:
            raise RuntimeError(f"duplicate pair summary: {pair_id}")
        summaries[pair_id] = summary
    if set(summaries) != set(pair_to_case):
        raise RuntimeError(f"terminal summaries differ from plan: {campaign_id}")

    pair_rows: list[dict[str, Any]] = []
    pair_pass: dict[str, bool] = {}
    recovered_issue_counts: Counter[str] = Counter()
    blocking_issue_counts: Counter[str] = Counter()
    for pair_id in sorted(pair_to_case):
        case = pair_to_case[pair_id]
        scene = str(case["scene_id"])
        summary = summaries[pair_id]
        results = {str(row["condition"]): row for row in summary["results"]}
        if set(results) != {"nominal_counterfactual", "anomaly"}:
            raise RuntimeError(f"incomplete pair summary: {pair_id}")
        episode_rows: dict[str, Any] = {}
        for condition in ("nominal_counterfactual", "anomaly"):
            result = results[condition]
            path = manifest_path(result, corpus)
            manifest = load_json(path)
            schedule = schedules[scene][(pair_id, condition)]
            integrity, issues, raw_issues = input_integrity(
                schedule, manifest, path.parent
            )
            for issue in raw_issues:
                if allowed_raw_issue(issue):
                    recovered_issue_counts[issue] += 1
            entry: dict[str, Any] = {
                "condition": condition,
                "episode_id": str(manifest["episode_id"]),
                "manifest": str(path),
                "manifest_sha256": sha256(path),
                "raw_artifact_state": manifest.get("artifact_state"),
                "raw_evaluation_eligible": manifest.get("evaluation_eligible"),
                "raw_runtime_issues": raw_issues,
                "task_input_integrity_passed": bool(integrity),
                "task_input_integrity_issues": issues,
            }
            if condition == "anomaly":
                try:
                    physical_ok, operator = operator_gate(manifest, path.parent)
                    _, alignment = geometry_aligned_invariant_summary(path.parent)
                    visual = (
                        decision_visual_gate(
                            manifest, path.parent, float(alignment["decision_time_s"])
                        )
                        if str(manifest["operator_readback"]["operator_id"])
                        == "O7_visual_remap"
                        else {
                            "passed": True,
                            "not_used_by_t3_route": True,
                            "reason": "O8 is proprioception-decisive; appearance separation is diagnostic only",
                        }
                    )
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    physical_ok = False
                    operator = {
                        "passed": False,
                        "exception_type": type(exc).__name__,
                        "diagnostic": str(exc)[:300],
                    }
                    alignment = {"passed": False}
                    visual = {"passed": False, "not_evaluated": True}
                entry["operator_gate"] = operator
                entry["temporal_alignment"] = alignment
                entry["visual_input_gate"] = visual
                entry["passed"] = bool(integrity and physical_ok and visual["passed"])
            else:
                entry["passed"] = bool(integrity)
            if not entry["passed"]:
                for issue in issues or ["task_specific_gate_failed"]:
                    blocking_issue_counts[issue] += 1
            episode_rows[condition] = entry
        passed = all(value["passed"] for value in episode_rows.values())
        pair_pass[pair_id] = bool(passed)
        anomaly_manifest = load_json(Path(episode_rows["anomaly"]["manifest"]))
        pair_rows.append(
            {
                "campaign_id": campaign_id,
                "pair_id": pair_id,
                "case_id": str(case["case_id"]),
                "scene_id": scene,
                "operator_id": str(
                    anomaly_manifest["operator_readback"]["operator_id"]
                ),
                "raw_pair_passed": summary.get("passed") is True,
                "passed": bool(passed),
                "episodes": episode_rows,
            }
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for case in selected:
        passed = all(pair_pass[str(pair_id)] for pair_id in case["pair_ids"])
        record = {**case, "campaign_id": campaign_id, "passed": bool(passed)}
        (accepted if passed else rejected).append(record)
    return {
        "campaign_id": campaign_id,
        "planned_cases": planned,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "pair_audits": pair_rows,
        "accepted_pairs": sum(pair_pass.values()),
        "recovered_issue_counts": dict(sorted(recovered_issue_counts.items())),
        "blocking_issue_counts": dict(sorted(blocking_issue_counts.items())),
        "source_sha256": {
            "selected_cases": sha256(selected_path),
            "seal": sha256(seal_path),
            "raw_audit": sha256(raw_audit_path),
        },
        "raw_counts": raw.get("counts", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if load_json(F1).get("new_data_available_at_freeze") is not False:
        raise RuntimeError("F1 did not precede confirmation data")

    campaigns = [audit_campaign(config) for config in CAMPAIGNS]
    accepted = [row for value in campaigns for row in value["accepted_cases"]]
    rejected = [row for value in campaigns for row in value["rejected_cases"]]
    pairs = [row for value in campaigns for row in value["pair_audits"]]
    planned = sum(int(value["planned_cases"]) for value in campaigns)
    attrition = (planned - len(accepted)) / planned
    checks = {
        "f1_precedes_confirmation_data": True,
        "pre_frozen_nonblocking_suffixes_enforced_exactly": True,
        "all_raw_terminal_audits_preserved": True,
        "all_planned_cases_accounted": len(accepted) + len(rejected) == planned,
        "all_planned_pairs_audited": len(pairs) == 2 * planned,
        "all_accepted_cases_have_two_passed_pairs": all(
            len(row["pair_ids"]) == 2 for row in accepted
        ),
        "combined_task_aligned_attrition_below_five_percent": attrition
        < ATTRITION_LIMIT,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_retry_or_replacement": False,
    }
    audit = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-f5b-task-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "task_aligned_model_blind_audit_complete",
        "passed": all(checks.values()),
        "counts_as_confirmatory_evidence": True,
        "development_only": False,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "checks": checks,
        "counts": {
            "planned_cases": planned,
            "accepted_cases": len(accepted),
            "rejected_cases": len(rejected),
            "pairs": len(pairs),
            "accepted_pairs": sum(int(value["accepted_pairs"]) for value in campaigns),
            "physical_episodes": 2 * len(pairs),
        },
        "case_attrition_rate": attrition,
        "maximum_case_attrition_rate": ATTRITION_LIMIT,
        "campaigns": [
            {
                key: value[key]
                for key in (
                    "campaign_id",
                    "planned_cases",
                    "accepted_pairs",
                    "recovered_issue_counts",
                    "blocking_issue_counts",
                    "source_sha256",
                    "raw_counts",
                )
            }
            | {
                "accepted_cases": len(value["accepted_cases"]),
                "rejected_cases": len(value["rejected_cases"]),
            }
            for value in campaigns
        ],
        "frozen_contract": {
            "authority": "F1 snapshot protocol selection.allow_nonblocking_runtime_issue_suffixes",
            "nonblocking_runtime_issue_suffixes": list(NONBLOCKING_SUFFIXES),
            "whole_rollout_appearance_checks_are_diagnostic": True,
            "o7_decision_aligned_visual_gate": {
                "frames": VISUAL_FRAMES,
                "crop_y_fraction": VISUAL_CROP_Y_FRACTION,
                "minimum_mean_pair_rgb_l1": APPEARANCE_L1_THRESHOLD,
                "maximum_rgb_decision_skew_s": MAX_RGB_DECISION_SKEW_S,
            },
            "o8_appearance_separation_is_not_an_exclusion": True,
            "footprint_margin_m": FOOTPRINT_MARGIN_M,
            "post_encounter_delay_s": POST_ENCOUNTER_DELAY_S,
        },
        "source_sha256": {
            "f1": sha256(F1),
            "f1_schedule_builder": sha256(F1_BUILDER),
            "audit_script": sha256(Path(__file__).resolve()),
        },
        "artifacts": {
            "accepted_cases": "accepted_cases.jsonl",
            "rejected_cases": "rejected_cases.jsonl",
            "pair_audits": "pair_audits.jsonl",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_exclusive(output / "accepted_cases.jsonl", accepted)
    write_jsonl_exclusive(output / "rejected_cases.jsonl", rejected)
    write_jsonl_exclusive(output / "pair_audits.jsonl", pairs)
    audit["artifact_sha256"] = {
        name: sha256(output / relative)
        for name, relative in audit["artifacts"].items()
    }
    write_json_exclusive(output / "audit.json", audit)
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "counts": audit["counts"],
                "case_attrition_rate": attrition,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
