#!/usr/bin/env python3
"""Fail-closed F5b successor for terminal attempts without pair summaries.

F5b assumed that every terminal collection attempt emitted a pair summary.
The collector's frozen terminal contract also permits a first attempt to end
before that artifact is written (for example, timeout or process failure).
This successor keeps those pairs in the planned denominator, proves their
single terminal-attempt provenance, and marks them failed.  It does not retry,
replace, or admit any additional case and never opens model outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_kino_v4_confirmation_t3_f5b as base


ATTEMPTS = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"
    / "collection/attempts"
)


def _registry_sha256(values: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(sorted(values.items())), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attempt_receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": base.sha256(path),
        "state": str(value["state"]),
        "summary_exists": bool(value["summary_exists"]),
        "summary_passed": bool(value["summary_passed"]),
        "passed": bool(value["passed"]),
        "timed_out": bool(value["timed_out"]),
        "returncode": int(value["returncode"]),
        "scientific_collection_attempt_index": int(
            value["scientific_collection_attempt_index"]
        ),
        "result_dependent_retry": bool(value["result_dependent_retry"]),
        "retry_authorized": bool(value["retry_authorized"]),
        "model_prediction_truth_key_or_score_read": bool(
            value["model_prediction_truth_key_or_score_read"]
        ),
    }


def _load_terminal_attempts(
    directory: Path | None,
    pair_to_case: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    campaign_id: str,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], str | None]:
    planned = set(pair_to_case)
    if directory is None:
        if set(summaries) != planned:
            raise RuntimeError(
                f"terminal summaries differ from plan without attempt registry: {campaign_id}"
            )
        return {}, None
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    attempts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        value = base.load_json(path)
        pair_id = str(value.get("pair_id", ""))
        if not pair_id or pair_id in attempts:
            raise RuntimeError(f"invalid or duplicate terminal attempt: {path}")
        attempts[pair_id] = (path.resolve(), value)
    if set(attempts) != planned:
        raise RuntimeError(f"terminal attempts differ from plan: {campaign_id}")
    if not set(summaries).issubset(planned):
        raise RuntimeError(f"unexpected pair summary outside plan: {campaign_id}")

    hashes: dict[str, str] = {}
    for pair_id in sorted(planned):
        path, value = attempts[pair_id]
        case = pair_to_case[pair_id]
        summary_exists = pair_id in summaries
        checks = (
            str(value.get("pair_id")) == pair_id,
            path.stem == pair_id,
            str(value.get("scene_id")) == str(case["scene_id"]),
            value.get("state") == "terminal",
            int(value.get("scientific_collection_attempt_index", -1)) == 1,
            value.get("result_dependent_retry") is False,
            value.get("retry_authorized") is False,
            value.get("model_prediction_truth_key_or_score_read") is False,
            value.get("summary_exists") is summary_exists,
        )
        if not all(checks):
            raise RuntimeError(f"invalid frozen terminal attempt: {pair_id}")
        if summary_exists:
            expected_pass = summaries[pair_id].get("passed") is True
            if (
                value.get("summary_passed") is not expected_pass
                or value.get("passed") is not expected_pass
            ):
                raise RuntimeError(f"attempt/summary result mismatch: {pair_id}")
        elif (
            value.get("summary_passed") is not False
            or value.get("passed") is not False
            or int(value.get("returncode", 0)) == 0
        ):
            raise RuntimeError(f"summary-free attempt is not a failure: {pair_id}")
        hashes[pair_id] = base.sha256(path)
    return attempts, _registry_sha256(hashes)


def audit_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    campaign_id = str(config["id"])
    selected_path = Path(config["selected"]).resolve()
    seal_path = Path(config["seal"]).resolve()
    raw_audit_path = Path(config["raw_audit"]).resolve()
    corpus = Path(config["corpus"]).resolve()
    attempt_directory = (
        Path(config["terminal_attempts"]).resolve()
        if config.get("terminal_attempts") is not None
        else None
    )
    seal = base.load_json(seal_path)
    raw = base.load_json(raw_audit_path)
    selected = base.load_jsonl(selected_path)
    planned = int(seal["counts"]["planned_cases"])
    if (
        seal.get("passed") is not True
        or seal.get("model_prediction_truth_key_or_score_read") is not False
        or raw.get("checks", {}).get("all_pairs_terminal") is not True
        or raw.get("model_prediction_truth_key_or_score_read") is not False
        or raw.get("result_dependent_selection_or_retry") is not False
        or len(selected) != planned
        or base.sha256(selected_path) != seal.get("selected_cases_sha256")
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
                for row in base.load_jsonl(schedule_path)
            }
        for pair_id_value in case["pair_ids"]:
            pair_id = str(pair_id_value)
            if pair_id in pair_to_case:
                raise RuntimeError(f"duplicate pair id: {pair_id}")
            pair_to_case[pair_id] = case

    summaries: dict[str, dict[str, Any]] = {}
    for path in corpus.glob("*/pair_summaries/*.json"):
        summary = base.load_json(path)
        pair_id = str(summary["counterfactual_group_id"])
        if pair_id in summaries:
            raise RuntimeError(f"duplicate pair summary: {pair_id}")
        summaries[pair_id] = summary
    if not set(summaries).issubset(pair_to_case):
        raise RuntimeError(f"unexpected pair summary outside plan: {campaign_id}")
    attempts, attempt_registry_sha256 = _load_terminal_attempts(
        attempt_directory, pair_to_case, summaries, campaign_id
    )

    pair_rows: list[dict[str, Any]] = []
    pair_pass: dict[str, bool] = {}
    recovered_issue_counts: Counter[str] = Counter()
    blocking_issue_counts: Counter[str] = Counter()
    for pair_id in sorted(pair_to_case):
        case = pair_to_case[pair_id]
        scene = str(case["scene_id"])
        attempt = attempts.get(pair_id)
        attempt_receipt = (
            _attempt_receipt(attempt[0], attempt[1]) if attempt is not None else None
        )
        if pair_id not in summaries:
            blocking_issue_counts["terminal_attempt_without_pair_summary"] += 1
            anomaly_schedule = schedules[scene][(pair_id, "anomaly")]
            pair_pass[pair_id] = False
            pair_rows.append(
                {
                    "campaign_id": campaign_id,
                    "pair_id": pair_id,
                    "case_id": str(case["case_id"]),
                    "scene_id": scene,
                    "operator_id": str(anomaly_schedule["target_operator"]),
                    "raw_pair_passed": False,
                    "passed": False,
                    "failure_reason": "terminal_attempt_without_pair_summary",
                    "terminal_attempt": attempt_receipt,
                    "episodes": {},
                }
            )
            continue

        summary = summaries[pair_id]
        results = {str(row["condition"]): row for row in summary["results"]}
        if set(results) != {"nominal_counterfactual", "anomaly"}:
            raise RuntimeError(f"incomplete pair summary: {pair_id}")
        episode_rows: dict[str, Any] = {}
        for condition in ("nominal_counterfactual", "anomaly"):
            result = results[condition]
            path = base.manifest_path(result, corpus)
            manifest = base.load_json(path)
            schedule = schedules[scene][(pair_id, condition)]
            integrity, issues, raw_issues = base.input_integrity(
                schedule, manifest, path.parent
            )
            for issue in raw_issues:
                if base.allowed_raw_issue(issue):
                    recovered_issue_counts[issue] += 1
            entry: dict[str, Any] = {
                "condition": condition,
                "episode_id": str(manifest["episode_id"]),
                "manifest": str(path),
                "manifest_sha256": base.sha256(path),
                "raw_artifact_state": manifest.get("artifact_state"),
                "raw_evaluation_eligible": manifest.get("evaluation_eligible"),
                "raw_runtime_issues": raw_issues,
                "task_input_integrity_passed": bool(integrity),
                "task_input_integrity_issues": issues,
            }
            if condition == "anomaly":
                try:
                    physical_ok, operator = base.operator_gate(manifest, path.parent)
                    _, alignment = base.geometry_aligned_invariant_summary(path.parent)
                    visual = (
                        base.decision_visual_gate(
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
        anomaly_manifest = base.load_json(Path(episode_rows["anomaly"]["manifest"]))
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
                "terminal_attempt": attempt_receipt,
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
            "selected_cases": base.sha256(selected_path),
            "seal": base.sha256(seal_path),
            "raw_audit": base.sha256(raw_audit_path),
            "terminal_attempt_registry": attempt_registry_sha256,
        },
        "raw_counts": raw.get("counts", {}),
    }


def main() -> int:
    campaigns: list[dict[str, Any]] = []
    for config in base.CAMPAIGNS:
        value = dict(config)
        if value["id"] == "extension_f4l":
            value["terminal_attempts"] = ATTEMPTS
        campaigns.append(value)
    base.CAMPAIGNS = tuple(campaigns)
    base.audit_campaign = audit_campaign
    base.__file__ = str(Path(__file__).resolve())
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
