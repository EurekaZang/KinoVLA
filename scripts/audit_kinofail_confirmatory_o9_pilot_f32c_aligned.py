#!/usr/bin/env python3
"""Audit F32c at the anomaly-matched diagnostic horizon, model-blind."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_pair_alignment import audit_matched_horizon
from scripts.run_kinofail_t3_replenishment_f28 import atomic_json, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32c"
ALLOWED_VISUAL_ISSUES = ("appearance_effect_too_small", "rgb_spatial_contrast_too_low")


def allowed(values: list[Any]) -> bool:
    return all(any(str(value).endswith(suffix) for suffix in ALLOWED_VISUAL_ISSUES) for value in values)


def main() -> int:
    old_audit = read_json(OUTPUT / "final_audit.json")
    if (
        old_audit.get("status") != "final"
        or old_audit.get("counts", {}).get("terminal_pairs") != 12
        or old_audit.get("model_prediction_feature_label_or_score_read") is not False
    ):
        raise RuntimeError("F32c terminal model-blind audit is incomplete")
    rows: list[dict[str, Any]] = []
    for attempt_path in sorted((OUTPUT / "attempts").glob("*.json")):
        attempt = read_json(attempt_path)
        summary_path = Path(str(attempt.get("evidence", {}).get("summary", "")))
        summary = read_json(summary_path)
        by_condition = {str(row["condition"]): row for row in summary["results"]}
        if set(by_condition) != {"nominal_counterfactual", "anomaly"}:
            raise RuntimeError(f"incomplete F32c pair: {attempt_path.stem}")
        manifests = {
            condition: Path(str(result["manifest"])) for condition, result in by_condition.items()
        }
        runtime_issues: dict[str, list[Any]] = {}
        for condition, manifest_path in manifests.items():
            manifest = read_json(manifest_path)
            runtime_issues[condition] = list(manifest.get("runtime_validation", {}).get("issues", []))
        alignment = audit_matched_horizon(manifests["anomaly"], manifests["nominal_counterfactual"])
        runtime_pass = all(allowed(values) for values in runtime_issues.values())
        semantic_pass = bool(
            attempt.get("evidence", {}).get("anomaly_semantic", {}).get("passed")
        )
        rows.append(
            {
                "pair_id": attempt_path.stem,
                "passed": semantic_pass and runtime_pass and alignment["passed"],
                "strict_direct_semantic_passed": semantic_pass,
                "allowed_runtime_issues_only": runtime_pass,
                "matched_horizon": alignment,
                "full_episode_nominal_fallen_reported_only": attempt.get("evidence", {}).get("nominal_fallen"),
                "full_episode_nominal_max_route_deviation_m_reported_only": attempt.get("evidence", {}).get("nominal_max_route_deviation_m"),
                "runtime_issues": runtime_issues,
            }
        )
    passed = sum(bool(row["passed"]) for row in rows)
    audit = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32c-aligned-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final_developmental_pilot_interpretation",
        "passed": len(rows) == 12 and passed == 12,
        "model_prediction_feature_label_outcome_or_score_read": False,
        "pilot_pairs_permanently_excluded_from_final_confirmation": True,
        "alignment_rule": "audit nominal only through the anomaly-matched diagnostic decision time; retain later outcomes as non-admission metadata",
        "rule_frozen_for_f33_before_f33_collection": True,
        "counts": {
            "terminal_pairs": len(rows),
            "strict_semantic_and_matched_horizon_passed_pairs": passed,
            "rejected_pairs": len(rows) - passed,
        },
        "results": rows,
    }
    atomic_json(OUTPUT / "aligned_final_audit.json", audit)
    print(json.dumps({"passed": audit["passed"], "counts": audit["counts"]}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
