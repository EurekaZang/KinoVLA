#!/usr/bin/env python3
"""Audit F38-S1 against the frozen T3 decision-input semantics.

The generic runtime validator measures appearance separation over the complete
rollout and measures regional exposure at the base centre.  Neither quantity
matches the frozen C2-v5 T3 input:

* proprioception ends 0.30 s after the first Go2-footprint encounter; and
* vision is the five synchronized frames nearest that decision time, cropped
  to the lower 55 percent of the front-camera image.

This audit does not change the raw F38-S1 result.  It independently rechecks
the schedule binding and every artifact used by T3, substitutes the
already-frozen 0.35 m footprint definition for O7/O8 exposure, and applies the
unchanged 0.015 RGB-L1 threshold to the exact visual tensor consumed by T3.
No model, truth key, prediction, score, or outcome-strength statistic is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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


DEFAULT_SELECTED = (
    ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal_s1/selected_cases.jsonl"
)
DEFAULT_SEAL = (
    ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal_s1/seal_manifest.json"
)
DEFAULT_RAW_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"
DEFAULT_CORPUS = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal_s1/corpus"
)
DEFAULT_OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit"
PLANNED_CASES = 1_500
MINIMUM_ACCEPTED_CASES = 1_426
APPEARANCE_L1_THRESHOLD = 0.015
VISUAL_CROP_Y_FRACTION = 0.45
VISUAL_FRAMES = 5
MAX_RGB_DECISION_SKEW_S = 0.051
VIEWS = ("primary", "swap_01", "swap_02")
NON_MODEL_ARTIFACT_TOKENS = ("prediction", "truth_key", "score")
ALLOWED_RAW_RUNTIME_ISSUES = {
    "rgb_view_swap_01_appearance_effect_too_small",
    "rgb_view_swap_02_appearance_effect_too_small",
    "rgb_view_swap_02_rgb_spatial_contrast_too_low",
    "operator_local_qa_failed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def canonical_json(value: Any) -> str:
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
            stream.write(canonical_json(row) + "\n")


def _manifest_path(result: Mapping[str, Any], corpus: Path) -> Path:
    path = Path(str(result["manifest"]))
    if not path.is_absolute():
        path = corpus / path
    resolved = path.resolve()
    if corpus.resolve() not in resolved.parents:
        raise RuntimeError(f"manifest escapes the frozen corpus: {path}")
    return resolved


def _footprint_encounter(manifest: Mapping[str, Any], episode_dir: Path) -> dict[str, Any]:
    region = manifest["geometry_readback"]["operator_region"]
    telemetry_path = episode_dir / manifest["artifacts"]["telemetry"]["path"]
    encounter_rows: list[dict[str, Any]] = []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        x, y = (float(value) for value in row["position_xy_m"])
        dx = max(abs(x - float(region["cx"])) - float(region["hx"]), 0.0)
        dy = max(abs(y - float(region["cy"])) - float(region["hy"]), 0.0)
        if math.hypot(dx, dy) <= FOOTPRINT_MARGIN_M:
            encounter_rows.append(row)
    if not encounter_rows:
        raise ValueError(f"no {FOOTPRINT_MARGIN_M:.2f} m footprint encounter")
    first = encounter_rows[0]
    return {
        "encounter_time_s": float(first["timestamp_s"]),
        "encounter_position_xy_m": [float(value) for value in first["position_xy_m"]],
        "footprint_margin_m": FOOTPRINT_MARGIN_M,
        "encounter_rows": len(encounter_rows),
    }


def _task_operator_gate(
    manifest: Mapping[str, Any], episode_dir: Path
) -> tuple[bool, dict[str, Any]]:
    readback = manifest["operator_readback"]
    operator = str(readback["operator_id"])
    telemetry = readback["telemetry"]
    encounter = _footprint_encounter(manifest, episode_dir)
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
        raise RuntimeError(f"unexpected F38 T3 operator: {operator}")
    return bool(passed), {
        "operator_id": operator,
        "passed": bool(passed),
        "raw_operator_qa_passed": readback.get("qa_passed") is True,
        "raw_base_center_exposure_steps": int(
            telemetry.get("scale_region_exposure_steps", 0)
        ),
        **encounter,
    }


def _task_input_integrity(
    schedule: Mapping[str, Any], manifest: dict[str, Any], episode_dir: Path
) -> tuple[bool, list[str]]:
    issues: list[str] = []
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
    raw_issues = set(manifest.get("runtime_validation", {}).get("issues", []))
    unexpected = sorted(raw_issues - ALLOWED_RAW_RUNTIME_ISSUES)
    issues.extend(f"unexpected_raw_issue:{value}" for value in unexpected)
    if (
        "operator_local_qa_failed" in raw_issues
        and not (
            manifest.get("operator_readback", {}).get("active") is True
            and manifest.get("operator_readback", {}).get("operator_id")
            == "O7_visual_remap"
        )
    ):
        issues.append("operator_local_warning_outside_o7_anomaly")
    for key in ("proprio", "telemetry"):
        artifact = manifest.get("artifacts", {}).get(key, {})
        path = episode_dir / str(artifact.get("path", ""))
        if not path.is_file():
            issues.append(f"{key}_missing")
        elif sha256(path) != artifact.get("sha256"):
            issues.append(f"{key}_hash_mismatch")
    if set(manifest.get("artifacts", {}).get("rgb_views", {})) != set(VIEWS):
        issues.append("rgb_view_set_mismatch")
    return not issues, issues


def _load_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256(path) != expected_sha256:
        raise ValueError(f"decision RGB hash mismatch: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _decision_aligned_visual_gate(
    manifest: Mapping[str, Any], episode_dir: Path, decision_time_s: float
) -> dict[str, Any]:
    views = manifest["artifacts"]["rgb_views"]
    if set(views) != set(VIEWS):
        raise ValueError(f"unexpected RGB views: {sorted(views)}")
    primary = views["primary"]
    if len(primary) < VISUAL_FRAMES:
        raise ValueError("too few primary RGB frames")
    times = np.asarray([float(row["timestamp_s"]) for row in primary])
    end_index = int(np.argmin(np.abs(times - decision_time_s)))
    end_skew = abs(float(times[end_index]) - decision_time_s)
    if end_skew > MAX_RGB_DECISION_SKEW_S or end_index + 1 < VISUAL_FRAMES:
        raise ValueError("decision-aligned RGB window unavailable")
    selected = list(range(end_index - VISUAL_FRAMES + 1, end_index + 1))
    for view_id in VIEWS[1:]:
        alternate = views[view_id]
        if len(alternate) != len(primary) or any(
            not math.isclose(
                float(primary[index]["timestamp_s"]),
                float(alternate[index]["timestamp_s"]),
                abs_tol=1.0e-9,
                rel_tol=0.0,
            )
            for index in selected
        ):
            raise ValueError(f"unsynchronized decision RGB: {view_id}")
    pair_l1: dict[str, float] = {}
    height: int | None = None
    selected_paths: dict[str, list[str]] = {view: [] for view in VIEWS}
    for view_id in VIEWS[1:]:
        differences: list[float] = []
        for index in selected:
            primary_relative = str(primary[index]["path"])
            alternate_relative = str(views[view_id][index]["path"])
            primary_rgb = _load_rgb(
                episode_dir / primary_relative, str(primary[index]["sha256"])
            )
            alternate_rgb = _load_rgb(
                episode_dir / alternate_relative,
                str(views[view_id][index]["sha256"]),
            )
            if primary_rgb.shape != alternate_rgb.shape:
                raise ValueError(f"decision RGB shape mismatch: {view_id}")
            if height is None:
                height = int(primary_rgb.shape[0])
            crop_start = int(math.floor(VISUAL_CROP_Y_FRACTION * primary_rgb.shape[0]))
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
            selected_paths["primary"].append(primary_relative)
            selected_paths[view_id].append(alternate_relative)
        pair_l1[view_id] = float(np.mean(differences))
    if height is None:
        raise ValueError("decision RGB images were not read")
    # Primary paths were appended once per alternate; retain one synchronized set.
    selected_paths["primary"] = selected_paths["primary"][:VISUAL_FRAMES]
    passed = all(
        pair_l1[view_id] >= APPEARANCE_L1_THRESHOLD for view_id in VIEWS[1:]
    )
    return {
        "passed": bool(passed),
        "decision_time_s": float(decision_time_s),
        "selected_frame_indices": selected,
        "selected_timestamps_s": [float(times[index]) for index in selected],
        "end_skew_s": float(end_skew),
        "frames": VISUAL_FRAMES,
        "crop_y_fraction": VISUAL_CROP_Y_FRACTION,
        "crop_y_start_px": int(math.floor(VISUAL_CROP_Y_FRACTION * height)),
        "mean_pair_rgb_l1": pair_l1,
        "minimum_mean_pair_rgb_l1": APPEARANCE_L1_THRESHOLD,
        "selected_paths": selected_paths,
    }


def _forbidden_model_artifacts() -> list[str]:
    roots = [
        ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2",
        ROOT / "outputs/kinofail_reconfirmation_v2",
        ROOT / "outputs/kinofail_t3_runin_f38_formal_s1",
    ]
    forbidden: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(token in name for token in NON_MODEL_ARTIFACT_TOKENS):
                forbidden.append(str(path))
    return sorted(set(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--raw-audit", type=Path, default=DEFAULT_RAW_AUDIT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected_path = args.selected.resolve()
    seal_path = args.seal.resolve()
    raw_audit_path = args.raw_audit.resolve()
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite task-aligned audit: {output}")
    seal = read_json(seal_path)
    raw_audit = read_json(raw_audit_path)
    selected = read_jsonl(selected_path)
    if (
        seal.get("passed") is not True
        or raw_audit.get("passed") is not False
        or raw_audit.get("model_prediction_truth_key_or_score_read") is not False
        or raw_audit.get("result_dependent_selection_or_retry") is not False
        or len(selected) != 1_485
        or int(seal.get("counts", {}).get("planned_cases", -1)) != PLANNED_CASES
    ):
        raise RuntimeError("F38-S1 predecessor state is not the sealed model-blind failure")
    forbidden = _forbidden_model_artifacts()
    if forbidden:
        raise RuntimeError(f"model-result artifact exists before task audit: {forbidden[:5]}")

    schedules: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    pair_to_case: dict[str, dict[str, Any]] = {}
    for case in selected:
        scene = str(case["scene_id"])
        schedule_path = (ROOT / str(case["schedule"])).resolve()
        if scene not in schedules:
            rows = read_jsonl(schedule_path)
            schedules[scene] = {
                (str(row["counterfactual_group_id"]), str(row["condition"])): row
                for row in rows
            }
        for pair_id in case["pair_ids"]:
            if pair_id in pair_to_case:
                raise RuntimeError(f"duplicate selected pair: {pair_id}")
            pair_to_case[str(pair_id)] = case

    summaries: dict[str, dict[str, Any]] = {}
    for path in corpus.glob("*/pair_summaries/*.json"):
        summary = read_json(path)
        pair_id = str(summary["counterfactual_group_id"])
        if pair_id in summaries:
            raise RuntimeError(f"duplicate pair summary: {pair_id}")
        summaries[pair_id] = summary
    if set(summaries) != set(pair_to_case):
        raise RuntimeError("F38-S1 pair summaries do not equal the selected pair set")

    pair_audits: list[dict[str, Any]] = []
    pair_pass: dict[str, bool] = {}
    raw_center_fail_recovered = 0
    for pair_id in sorted(pair_to_case):
        case = pair_to_case[pair_id]
        scene = str(case["scene_id"])
        summary = summaries[pair_id]
        results = {str(row["condition"]): row for row in summary["results"]}
        if set(results) != {"nominal_counterfactual", "anomaly"}:
            raise RuntimeError(f"incomplete pair summary: {pair_id}")
        episode_audits: dict[str, Any] = {}
        for condition in ("nominal_counterfactual", "anomaly"):
            result = results[condition]
            manifest_path = _manifest_path(result, corpus)
            manifest = read_json(manifest_path)
            schedule = schedules[scene][(pair_id, condition)]
            generic_integrity_passed, generic_issues = _task_input_integrity(
                schedule, manifest, manifest_path.parent
            )
            entry: dict[str, Any] = {
                "condition": condition,
                "episode_id": str(manifest["episode_id"]),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "raw_artifact_state": manifest.get("artifact_state"),
                "raw_evaluation_eligible": manifest.get("evaluation_eligible"),
                "raw_runtime_issues": list(
                    manifest.get("runtime_validation", {}).get("issues", [])
                ),
                "task_input_integrity_passed": generic_integrity_passed,
                "task_input_integrity_issues": generic_issues,
            }
            if condition == "anomaly":
                operator_passed, operator_audit = _task_operator_gate(
                    manifest, manifest_path.parent
                )
                _, alignment = geometry_aligned_invariant_summary(manifest_path.parent)
                entry["operator_gate"] = operator_audit
                entry["temporal_alignment"] = alignment
                if (
                    operator_passed
                    and manifest["operator_readback"].get("qa_passed") is not True
                ):
                    raw_center_fail_recovered += 1
                if str(manifest["operator_readback"]["operator_id"]) == "O7_visual_remap":
                    entry["visual_input_gate"] = _decision_aligned_visual_gate(
                        manifest, manifest_path.parent, float(alignment["decision_time_s"])
                    )
                else:
                    entry["visual_input_gate"] = {
                        "passed": True,
                        "not_used_by_t3": True,
                        "visual_source_operator": "O7_visual_remap",
                    }
                entry["passed"] = bool(
                    generic_integrity_passed
                    and operator_passed
                    and entry["visual_input_gate"]["passed"]
                )
            else:
                entry["passed"] = bool(generic_integrity_passed)
            episode_audits[condition] = entry
        passed = all(row["passed"] for row in episode_audits.values())
        pair_pass[pair_id] = passed
        anomaly_manifest = read_json(
            Path(episode_audits["anomaly"]["manifest"])
        )
        pair_audits.append(
            {
                "pair_id": pair_id,
                "case_id": str(case["case_id"]),
                "scene_id": scene,
                "operator_id": str(anomaly_manifest["operator_readback"]["operator_id"]),
                "raw_pair_passed": summary.get("passed") is True,
                "passed": bool(passed),
                "episodes": episode_audits,
            }
        )

    accepted_cases: list[dict[str, Any]] = []
    rejected_cases: list[dict[str, Any]] = []
    for case in selected:
        passed = all(pair_pass[str(pair_id)] for pair_id in case["pair_ids"])
        record = {**case, "passed": bool(passed)}
        (accepted_cases if passed else rejected_cases).append(record)
    attrition = 1.0 - len(accepted_cases) / PLANNED_CASES
    checks = {
        "raw_failure_preserved": raw_audit.get("passed") is False,
        "raw_failure_attrition_35_8667_percent": math.isclose(
            float(raw_audit["case_attrition_rate"]), 538 / PLANNED_CASES
        ),
        "no_model_prediction_truth_key_or_score_read": True,
        "no_result_dependent_retry": True,
        "all_selected_pairs_audited": len(pair_audits) == 2 * len(selected),
        "all_accepted_cases_have_two_passed_pairs": all(
            len(case["pair_ids"]) == 2
            and all(pair_pass[str(pair_id)] for pair_id in case["pair_ids"])
            for case in accepted_cases
        ),
        "task_aligned_case_gate": len(accepted_cases) >= MINIMUM_ACCEPTED_CASES,
        "strictly_below_five_percent_attrition": attrition < 0.05,
    }
    audit = {
        "schema_version": "kinofail.t3-runin-f38-s1-task-aligned-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "development_only": False,
        "counts_as_confirmatory_evidence": False,
        "status": "pre_prediction_protocol_amendment_requires_successor_seal",
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "checks": checks,
        "counts": {
            "planned_cases": PLANNED_CASES,
            "selected_cases": len(selected),
            "permanently_excluded_predecessor_cases": PLANNED_CASES - len(selected),
            "accepted_cases": len(accepted_cases),
            "rejected_selected_cases": len(rejected_cases),
            "accepted_pairs": sum(pair_pass.values()),
            "pairs": len(pair_audits),
            "raw_base_center_failures_recovered_by_frozen_footprint": raw_center_fail_recovered,
        },
        "case_attrition_rate": attrition,
        "minimum_accepted_cases": MINIMUM_ACCEPTED_CASES,
        "task_alignment_contract": {
            "basis": "exact frozen C2-v5 T3 model-input semantics",
            "footprint_margin_m": FOOTPRINT_MARGIN_M,
            "post_encounter_delay_s": POST_ENCOUNTER_DELAY_S,
            "visual_frames": VISUAL_FRAMES,
            "visual_crop_y_fraction": VISUAL_CROP_Y_FRACTION,
            "maximum_rgb_decision_skew_s": MAX_RGB_DECISION_SKEW_S,
            "minimum_mean_pair_rgb_l1": APPEARANCE_L1_THRESHOLD,
            "visual_source_operator": "O7_visual_remap",
            "o8_rgb_is_not_used_by_t3": True,
            "whole_rollout_l1_is_diagnostic_only": True,
            "base_center_exposure_is_diagnostic_only": True,
        },
        "predecessor": {
            "seal": str(seal_path),
            "seal_sha256": sha256(seal_path),
            "selected_cases": str(selected_path),
            "selected_cases_sha256": sha256(selected_path),
            "raw_final_audit": str(raw_audit_path),
            "raw_final_audit_sha256": sha256(raw_audit_path),
            "raw_passed": False,
            "raw_case_attrition_rate": float(raw_audit["case_attrition_rate"]),
        },
        "accepted_cases_path": "accepted_cases.jsonl",
        "rejected_cases_path": "rejected_cases.jsonl",
        "pair_audits_path": "pair_audits.jsonl",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_exclusive(output / "accepted_cases.jsonl", accepted_cases)
    write_jsonl_exclusive(output / "rejected_cases.jsonl", rejected_cases)
    write_jsonl_exclusive(output / "pair_audits.jsonl", pair_audits)
    audit["output_sha256"] = {
        "accepted_cases": sha256(output / "accepted_cases.jsonl"),
        "rejected_cases": sha256(output / "rejected_cases.jsonl"),
        "pair_audits": sha256(output / "pair_audits.jsonl"),
    }
    write_json_exclusive(output / "audit.json", audit)
    print(json.dumps({
        "passed": audit["passed"],
        "counts": audit["counts"],
        "case_attrition_rate": audit["case_attrition_rate"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
