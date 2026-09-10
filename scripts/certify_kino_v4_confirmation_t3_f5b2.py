#!/usr/bin/env python3
"""Certify F5b1 after correcting its pass-predicate serialization bug.

F5b1 completed every model-blind scientific check, but computed ``passed``
with ``all(checks.values())`` while two entries encoded prohibited-state
values as False.  This successor validates and byte-binds the complete F5b1
output, converts those two state values into positive predicates, and copies
the audited registries without changing membership.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"
    / "f5_task_aligned_audit"
)
OUTPUT = SOURCE.with_name("f5_task_aligned_audit_f5b2")
AUDITOR = ROOT / "scripts/audit_kino_v4_confirmation_t3_f5b1.py"
RESULT_ARTIFACTS = (
    ROOT / "outputs/freeze/kino_v4_confirmation_observations_f5c/observation_seal.json",
    ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d/feature_seal.json",
    ROOT / "outputs/eval/kino_v4_confirmation_v1_f5e_score_once/report.json",
    ROOT
    / "outputs/eval/kino_v4_confirmation_v1_f5e_score_once"
    / "publication_gate_audit.json",
)
POSITIVE_CHECKS = (
    "f1_precedes_confirmation_data",
    "pre_frozen_nonblocking_suffixes_enforced_exactly",
    "all_raw_terminal_audits_preserved",
    "all_planned_cases_accounted",
    "all_planned_pairs_audited",
    "all_accepted_cases_have_two_passed_pairs",
    "combined_task_aligned_attrition_below_five_percent",
)
NEGATIVE_STATE_FIELDS = (
    "model_prediction_truth_key_or_score_read",
    "result_dependent_retry_or_replacement",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def write_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    existing = [str(path) for path in RESULT_ARTIFACTS if path.exists()]
    if existing:
        raise RuntimeError(f"downstream result exists before F5b2: {existing}")
    source_audit_path = SOURCE / "audit.json"
    source = load(source_audit_path)
    checks = source.get("checks", {})
    if (
        source.get("status") != "task_aligned_model_blind_audit_complete"
        or source.get("passed") is not False
        or source.get("model_prediction_truth_key_or_score_read") is not False
        or source.get("result_dependent_selection_or_retry") is not False
        or set(checks) != set(POSITIVE_CHECKS) | set(NEGATIVE_STATE_FIELDS)
        or not all(checks.get(name) is True for name in POSITIVE_CHECKS)
        or not all(checks.get(name) is False for name in NEGATIVE_STATE_FIELDS)
        or source.get("source_sha256", {}).get("audit_script") != sha256(AUDITOR)
    ):
        raise RuntimeError("F5b1 does not have the exact pass-predicate bug signature")

    artifacts = source.get("artifacts", {})
    if set(artifacts) != {"accepted_cases", "rejected_cases", "pair_audits"}:
        raise RuntimeError("unexpected F5b1 artifact registry")
    paths = {name: SOURCE / relative for name, relative in artifacts.items()}
    for name, path in paths.items():
        if (
            not path.is_file()
            or sha256(path) != source.get("artifact_sha256", {}).get(name)
        ):
            raise RuntimeError(f"F5b1 artifact drift: {path}")
    accepted = jsonl(paths["accepted_cases"])
    rejected = jsonl(paths["rejected_cases"])
    pairs = jsonl(paths["pair_audits"])
    counts = source["counts"]
    planned = int(counts["planned_cases"])
    attrition = float(source["case_attrition_rate"])
    if (
        len(accepted) != int(counts["accepted_cases"])
        or len(rejected) != int(counts["rejected_cases"])
        or len(pairs) != int(counts["pairs"])
        or len(accepted) + len(rejected) != planned
        or sum(row.get("passed") is True for row in pairs)
        != int(counts["accepted_pairs"])
        or len(pairs) != 2 * planned
        or abs(attrition - len(rejected) / planned) > 1e-15
        or not attrition < float(source["maximum_case_attrition_rate"])
    ):
        raise RuntimeError("F5b1 audited registries or attrition do not reconcile")

    corrected_checks = {
        name: checks[name] for name in POSITIVE_CHECKS
    } | {
        "no_model_prediction_truth_key_or_score_read": (
            checks["model_prediction_truth_key_or_score_read"] is False
        ),
        "no_result_dependent_retry_or_replacement": (
            checks["result_dependent_retry_or_replacement"] is False
        ),
    }
    if not all(corrected_checks.values()):
        raise RuntimeError("corrected F5b2 predicates did not pass")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, source_path in paths.items():
        shutil.copyfile(source_path, OUTPUT / artifacts[name])
    audit = {
        **source,
        "schema_version": "kinofail.kino-v4-confirmation-t3-f5b2-certification.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "task_aligned_model_blind_audit_certified",
        "passed": True,
        "checks": corrected_checks,
        "certification": {
            "scope": "pass-predicate serialization only",
            "scientific_audit_recomputed": False,
            "audited_membership_changed": False,
            "threshold_or_gate_changed": False,
            "retry_replacement_or_deletion": False,
            "source_audit": str(source_audit_path.relative_to(ROOT)),
            "source_audit_sha256": sha256(source_audit_path),
            "source_auditor": str(AUDITOR.relative_to(ROOT)),
            "source_auditor_sha256": sha256(AUDITOR),
        },
        "source_sha256": {
            **source["source_sha256"],
            "predecessor_audit": sha256(source_audit_path),
            "certifier": sha256(Path(__file__).resolve()),
        },
    }
    audit["artifact_sha256"] = {
        name: sha256(OUTPUT / relative) for name, relative in artifacts.items()
    }
    write_exclusive(OUTPUT / "audit.json", audit)
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(OUTPUT),
                "planned_cases": planned,
                "accepted_cases": len(accepted),
                "rejected_cases": len(rejected),
                "case_attrition_rate": attrition,
                "model_prediction_truth_key_or_score_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
