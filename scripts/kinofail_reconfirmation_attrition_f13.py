#!/usr/bin/env python3
"""Model-blind launcher eligibility ledger for reconfirmation F13.

The physical snapshot selector intentionally judges immutable episode
manifests.  A launcher may nevertheless have declared an attempt terminal or
interrupted before the pair was admitted to the no-retry confirmatory sample.
F13 keeps those two layers separate: this module derives an eligibility ledger
only from the frozen schedule and launcher process state, before any model,
feature value, label, outcome, or score is read.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/attrition_ledgers_f13"
)
F13 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f13_launcher_eligibility_amendment1/"
    "amendment_manifest.json"
)


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
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise TypeError(path)
    return values


def _schedule_design(
    schedule_path: Path,
) -> tuple[str, list[str], dict[str, int]]:
    rows = read_jsonl(schedule_path)
    if not rows:
        raise RuntimeError("launcher ledger schedule is empty")
    scene_ids = {str(row["scene_family"]) for row in rows}
    if len(scene_ids) != 1:
        raise RuntimeError("launcher ledger schedule spans multiple scenes")
    by_pair: dict[str, int] = {}
    for row in rows:
        pair_id = str(row["counterfactual_group_id"])
        by_pair[pair_id] = by_pair.get(pair_id, 0) + 1
    if set(by_pair.values()) != {2}:
        raise RuntimeError("launcher ledger requires exactly two rows per pair")
    return next(iter(scene_ids)), sorted(by_pair), by_pair


def _attempt_is_eligible(attempt: dict[str, Any]) -> bool:
    # `passed` is deliberately not part of this operational gate.  The
    # pre-existing F5 O4 certificate is applied by the frozen physical
    # snapshot core.  A zero-return collector with a sealed summary reached
    # the physical-validation layer; interrupted or nonzero-return processes
    # did not.
    return (
        attempt.get("state") == "terminal"
        and int(attempt.get("returncode", -1)) == 0
        and attempt.get("summary_exists") is True
    )


def _attrition_reason(attempt: dict[str, Any]) -> str:
    if attempt.get("state") == "started":
        return "interrupted_started_without_retry"
    if attempt.get("state") == "terminal":
        return (
            "terminal_nonzero_or_missing_summary_without_retry:"
            f"returncode={attempt.get('returncode')}:"
            f"summary_exists={attempt.get('summary_exists')}"
        )
    return f"invalid_launcher_state_without_retry:{attempt.get('state')}"


def build_ledger(
    *,
    schedule_path: Path,
    launcher_audit_dir: Path,
    battery: str,
    output_path: Path,
) -> dict[str, Any]:
    schedule_path = schedule_path.resolve()
    launcher_audit_dir = launcher_audit_dir.resolve()
    scene_id, planned_pair_ids, _ = _schedule_design(schedule_path)
    planned = set(planned_pair_ids)
    audit_paths = sorted(
        launcher_audit_dir.glob(
            f"{battery}*_partition_*_of_*.json"
        )
    )
    if not audit_paths:
        raise RuntimeError(
            f"no launcher audits found for {scene_id}/{battery}"
        )
    attempt_by_pair: dict[str, dict[str, Any]] = {}
    attempt_source: dict[str, Path] = {}
    sources = []
    for audit_path in audit_paths:
        audit = read_json(audit_path)
        if str(audit.get("scene_id")) != scene_id:
            raise RuntimeError(f"launcher audit scene mismatch: {audit_path}")
        if str(audit.get("battery")) != battery:
            raise RuntimeError(
                f"launcher audit battery mismatch: {audit_path}"
            )
        sources.append(
            {
                "path": str(audit_path),
                "sha256": sha256(audit_path),
            }
        )
        for attempt in audit.get("attempts", []):
            pair_id = str(attempt["counterfactual_group_id"])
            if pair_id not in planned:
                raise RuntimeError(
                    f"launcher audit contains an unscheduled pair: {pair_id}"
                )
            if pair_id in attempt_by_pair:
                raise RuntimeError(
                    "result-dependent reexecution or duplicate launcher "
                    f"attempt detected: {pair_id}: "
                    f"{attempt_source[pair_id]} and {audit_path}"
                )
            attempt_by_pair[pair_id] = dict(attempt)
            attempt_source[pair_id] = audit_path
    missing = sorted(planned - set(attempt_by_pair))
    if missing:
        raise RuntimeError(
            f"launcher ledger has {len(missing)} unaccounted scheduled pairs"
        )
    eligible = sorted(
        pair_id
        for pair_id, attempt in attempt_by_pair.items()
        if _attempt_is_eligible(attempt)
    )
    attrition = [
        {
            "counterfactual_group_id": pair_id,
            "reason": _attrition_reason(attempt_by_pair[pair_id]),
            "launcher_audit": str(attempt_source[pair_id]),
            "launcher_attempt": attempt_by_pair[pair_id],
        }
        for pair_id in sorted(planned - set(eligible))
    ]
    f13_sha256 = sha256(F13) if F13.is_file() else None
    ledger = {
        "schema_version": (
            "kinofail.reconfirmation-f13-launcher-eligibility-ledger.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": True,
        "scene_id": scene_id,
        "battery": battery,
        "selection_basis": (
            "launcher process terminality, return code, and sealed summary "
            "existence only"
        ),
        "model_feature_label_outcome_or_score_read": False,
        "result_dependent_retry_or_selection": False,
        "scientific_sample_selection_rule_changed": False,
        "operational_amendment": str(F13),
        "operational_amendment_sha256": f13_sha256,
        "source_schedule": str(schedule_path),
        "source_schedule_sha256": sha256(schedule_path),
        "source_launcher_audits": sources,
        "counts": {
            "planned_pairs": len(planned_pair_ids),
            "eligible_pairs": len(eligible),
            "attrited_pairs": len(attrition),
            "launcher_attempt_records": len(attempt_by_pair),
        },
        "eligible_pair_ids": eligible,
        "attrition": attrition,
        "checks": {
            "every_scheduled_pair_has_exactly_one_attempt": (
                len(attempt_by_pair) == len(planned_pair_ids)
            ),
            "eligible_and_attrited_partition_schedule": (
                len(eligible) + len(attrition) == len(planned_pair_ids)
            ),
            "no_model_feature_label_outcome_or_score_read": True,
            "no_result_dependent_retry": True,
        },
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = read_json(output_path)
        for key in (
            "schema_version",
            "status",
            "passed",
            "scene_id",
            "battery",
            "source_schedule_sha256",
            "eligible_pair_ids",
            "attrition",
        ):
            if existing.get(key) != ledger.get(key):
                raise RuntimeError(
                    f"existing F13 ledger differs at {key}: {output_path}"
                )
        return existing
    output_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger


def ledger_path(scene_id: str, battery: str) -> Path:
    return LEDGER_ROOT / scene_id / f"{battery}.json"


def load_ledger(
    *, scene_id: str, battery: str
) -> tuple[dict[str, Any], Path]:
    path = ledger_path(scene_id, battery)
    ledger = read_json(path)
    if (
        ledger.get("passed") is not True
        or ledger.get("status") != "complete"
        or ledger.get("scene_id") != scene_id
        or ledger.get("battery") != battery
        or ledger.get("model_feature_label_outcome_or_score_read")
        is not False
        or ledger.get("result_dependent_retry_or_selection") is not False
        or ledger.get("checks", {}).get(
            "every_scheduled_pair_has_exactly_one_attempt"
        )
        is not True
        or ledger.get("checks", {}).get(
            "eligible_and_attrited_partition_schedule"
        )
        is not True
    ):
        raise RuntimeError(f"invalid F13 launcher ledger: {path}")
    return ledger, path


def build_for_protocol(
    *,
    protocol: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], Path] | None:
    protocol_id = str(protocol.get("protocol_id", ""))
    if not protocol_id.startswith("reconfirmation-v2-"):
        return None
    schedule_path = Path(str(protocol["source_schedule"]))
    if not schedule_path.is_absolute():
        schedule_path = repo_root / schedule_path
    corpus_root = Path(str(protocol["source_corpus_root"]))
    if not corpus_root.is_absolute():
        corpus_root = repo_root / corpus_root
    scene_id, _, _ = _schedule_design(schedule_path)
    battery = (
        "c2_t3"
        if "c2_t3" in schedule_path.parts
        or "-t3-" in protocol_id
        else "scale"
    )
    output_path = ledger_path(scene_id, battery)
    ledger = build_ledger(
        schedule_path=schedule_path,
        launcher_audit_dir=corpus_root / "launcher_audits",
        battery=battery,
        output_path=output_path,
    )
    return ledger, output_path

