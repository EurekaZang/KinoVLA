#!/usr/bin/env python3
"""Strict, telemetry-only O9 audit for accepted F25 replacement pairs.

Version 1 established whether any base contact was present.  This audit uses
the frozen semantic contract in :mod:`kino_vla.data.o9_semantics`: sustained
base/belly loading, a non-impulsive duty cycle, and partial foot unloading are
all required.  It never reads a model feature, prediction, label, or score.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import DEFAULT_THRESHOLDS, evaluate_o9_high_centering
from scripts.audit_kinofail_expanded_o9_semantics_v1 import (
    CORPUS,
    OVERLAY,
    ROOT,
    read_json,
    read_jsonl,
    sha256,
)


OUTPUT = (
    ROOT
    / "outputs/eval/unified_moe_v3_replenishment_f25/"
    "o9_semantic_audit_expanded_v2_strict.json"
)


def _episode_semantics(manifest_path: Path) -> dict[str, Any]:
    telemetry_path = manifest_path.parent / "privileged.jsonl"
    rows = read_jsonl(telemetry_path)
    operator_rows = [row.get("operator", {}) for row in rows]
    evaluations = [evaluate_o9_high_centering(row) for row in operator_rows]
    passing = [row for row in evaluations if row["passed"]]
    final = evaluations[-1] if evaluations else evaluate_o9_high_centering({})
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "telemetry": str(telemetry_path),
        "telemetry_sha256": sha256(telemetry_path),
        "operator_enabled_observed": any(
            row.get("enabled") is True for row in operator_rows
        ),
        "operator_disabled_throughout": all(
            row.get("enabled") is not True for row in operator_rows
        ),
        "strict_semantic_passed": bool(passing),
        "first_passing_frame_index": (
            next(i for i, row in enumerate(evaluations) if row["passed"])
            if passing
            else None
        ),
        "final_evaluation": final,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    overlay_audit_path = OVERLAY / "overlay_audit.json"
    overlay_audit = read_json(overlay_audit_path)
    if (
        overlay_audit.get("passed") is not True
        or overlay_audit.get("model_prediction_or_score_read") is not False
    ):
        raise RuntimeError("accepted Scale overlay is invalid")

    records: list[dict[str, Any]] = []
    record_paths = sorted(
        (OVERLAY / "shards").glob(
            "*/scale/accepted_snapshots/snapshot_records.jsonl"
        )
    )
    for path in record_paths:
        records.extend(read_jsonl(path))
    primary = {
        (str(row["counterfactual_group_id"]), str(row["condition"])): row
        for row in records
        if row["target_operator"] == "O9_high_centering"
        and row["appearance_intervention_id"] == "primary"
    }
    group_ids = sorted({group_id for group_id, _ in primary})
    expected_conditions = {"nominal_counterfactual", "anomaly"}
    if len(group_ids) != 24 or any(
        {condition for candidate, condition in primary if candidate == group_id}
        != expected_conditions
        for group_id in group_ids
    ):
        raise RuntimeError("accepted F25 O9 registry is incomplete")

    manifest_index = {path.parent.name: path for path in CORPUS.rglob("manifest.json")}
    cases: list[dict[str, Any]] = []
    for group_id in group_ids:
        evidence: dict[str, Any] = {}
        for condition in sorted(expected_conditions):
            record = primary[(group_id, condition)]
            episode_id = str(record["source_replacement_physical_episode_id"])
            manifest_path = manifest_index.get(episode_id)
            if manifest_path is None:
                raise RuntimeError(f"source manifest missing: {episode_id}")
            if sha256(manifest_path) != record["source_manifest_sha256"]:
                raise RuntimeError(f"source manifest hash mismatch: {episode_id}")
            evidence[condition] = _episode_semantics(manifest_path)
        nominal = evidence["nominal_counterfactual"]
        anomaly = evidence["anomaly"]
        checks = {
            "nominal_operator_disabled_throughout": nominal[
                "operator_disabled_throughout"
            ],
            "nominal_does_not_pass_o9_semantics": not nominal[
                "strict_semantic_passed"
            ],
            "anomaly_operator_enabled": anomaly["operator_enabled_observed"],
            "anomaly_passes_frozen_strict_o9_semantics": anomaly[
                "strict_semantic_passed"
            ],
        }
        record = primary[(group_id, "anomaly")]
        cases.append(
            {
                "counterfactual_group_id": group_id,
                "scene_id": str(record["scene_family"]),
                "domain": str(record["domain"]),
                "severity_id": str(record["severity_id"]),
                "passed": all(checks.values()),
                "checks": checks,
                "evidence": evidence,
            }
        )

    passed = [row for row in cases if row["passed"]]
    audit = {
        "schema_version": "kinofail.expanded-o9-semantic-audit.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": len(passed) == len(cases),
        "model_prediction_feature_label_or_score_read": False,
        "semantic_contract": {
            "definition": (
                "sustained Go2 base/belly load on a transverse ridge with "
                "partial foot unloading; proximity and head/limb collision fail"
            ),
            "thresholds": DEFAULT_THRESHOLDS,
            "module": "kino_vla/data/o9_semantics.py",
            "module_sha256": sha256(ROOT / "kino_vla/data/o9_semantics.py"),
        },
        "counts": {
            "accepted_f25_o9_pairs": len(cases),
            "strictly_admitted_pairs": len(passed),
            "strictly_rejected_pairs": len(cases) - len(passed),
            "scenes": len({row["scene_id"] for row in cases}),
            "domains": len({row["domain"] for row in cases}),
            "admitted_by_severity": dict(
                sorted(Counter(row["severity_id"] for row in passed).items())
            ),
        },
        "disposition": (
            "Only strictly admitted pairs may be used as O9 evidence.  The "
            "expanded confirmation overlay will replace all legacy O9 slots "
            "because their direct-contact telemetry was pruned before this audit."
        ),
        "source_sha256": {
            "overlay_audit": sha256(overlay_audit_path),
            "snapshot_record_shards": {
                str(path.relative_to(ROOT)): sha256(path) for path in record_paths
            },
        },
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: audit[key] for key in ("status", "passed", "counts", "disposition")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
