#!/usr/bin/env python3
"""Seal the one-shot O4 phase-aware v6 development replay."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/data/kinofail_o4_phase_aware_v6_development_plan.json"
PLAN_SHA256 = "f219a58aba7fc7af0a29656fb6fc1035fd328167bf5359f236d25dcbf7101c35"
OUT = (
    ROOT
    / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2"
    / "o4_phase_aware_v6_development_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _hash_spec(spec: dict[str, Any]) -> bool:
    path = _resolve(str(spec["path"]))
    return path.is_file() and _sha256(path) == spec["sha256"]


def _expected_control(record: dict[str, Any], role: str) -> bool:
    checks = record.get("checks", {})
    visual = record.get("phase_aware_visuals", {})
    common = (
        record.get("adjudication_role") == "development"
        and checks.get("all_artifact_hashes_verified") is True
        and checks.get("phase_aware_visual_contract_passed") is True
        and visual.get("passed") is True
    )
    if role == "positive":
        return (
            common
            and checks.get("all_nonvisual_pair_checks_passed") is True
            and record.get("passed") is True
        )
    return (
        common
        and checks.get("all_nonvisual_pair_checks_passed") is False
        and record.get("passed") is False
    )


def main() -> None:
    plan = _json(PLAN)
    frozen_checks: dict[str, bool] = {
        "plan_hash": _sha256(PLAN) == PLAN_SHA256,
        "frozen_before_new_formal_collection": plan.get("status")
        == "frozen_after_development_replay_before_new_formal_collection",
        "old_batches_remain_sealed": plan["decision_rule"].get("old_batches_remain_sealed")
        is True,
        "new_formal_threshold_iteration_forbidden": plan["decision_rule"].get(
            "new_formal_threshold_iteration_allowed"
        )
        is False,
    }
    for name, spec in plan["frozen_implementation"].items():
        frozen_checks[f"implementation::{name}"] = _hash_spec(spec)

    controls: list[dict[str, Any]] = []
    for role, specs in (
        ("positive", plan["development_positive_controls"]),
        ("physical_negative", plan["physical_negative_controls"]),
    ):
        for spec in specs:
            path = _resolve(spec["path"])
            hash_ok = _hash_spec(spec)
            value = _json(path) if path.is_file() else {}
            expected = hash_ok and value.get("scene_id") == spec["scene_id"] and _expected_control(
                value, "positive" if role == "positive" else "negative"
            )
            controls.append(
                {
                    "role": role,
                    "scene_id": spec["scene_id"],
                    "path": str(path),
                    "sha256": _sha256(path) if path.is_file() else None,
                    "hash_ok": hash_ok,
                    "expected_behavior": expected,
                    "input_pair_passed": value.get("input_pair_passed"),
                    "old_visual_passed": value.get("input_visual_check_passed"),
                    "nonvisual_passed": value.get("checks", {}).get(
                        "all_nonvisual_pair_checks_passed"
                    ),
                    "phase_aware_visual_passed": value.get("phase_aware_visuals", {}).get(
                        "passed"
                    ),
                    "v6_passed": value.get("passed"),
                }
            )

    correct_count = sum(row["expected_behavior"] for row in controls)
    passed = (
        all(frozen_checks.values())
        and len(controls) == 6
        and correct_count == len(controls)
    )
    payload = {
        "schema_version": "kinofail.o4-phase-aware-development-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(PLAN), "sha256": _sha256(PLAN)},
        "passed": passed,
        "decision": "admit_for_new_heldout_batch" if passed else "reject",
        "frozen_checks": frozen_checks,
        "controls": controls,
        "summary": {
            "positive_controls_passed": sum(
                row["expected_behavior"] for row in controls if row["role"] == "positive"
            ),
            "positive_controls_total": sum(row["role"] == "positive" for row in controls),
            "physical_negative_controls_retained": sum(
                row["expected_behavior"]
                for row in controls
                if row["role"] == "physical_negative"
            ),
            "physical_negative_controls_total": sum(
                row["role"] == "physical_negative" for row in controls
            ),
        },
        "interpretation": (
            "Development-only evidence supports freezing v6 for a new held-out batch. No sealed "
            "v4/v5 result is relabeled, and the physical thresholds remain unchanged."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
