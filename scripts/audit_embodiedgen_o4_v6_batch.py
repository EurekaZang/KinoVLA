#!/usr/bin/env python3
"""Audit a preregistered O4 v6 batch without outcome-dependent exclusions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audit_embodiedgen_o4_corridor_v2_batch import (
    ROOT,
    _attempt as _v5_attempt,
    _hash_spec,
    _json,
    _resolve,
    _sha256,
)


def _v6_attempt(
    request: dict[str, Any], *, batch: dict[str, Any], scene_root: Path
) -> dict[str, Any]:
    """Extend the frozen corridor-v2 audit with the preregistered v6 adjudicator."""

    base = _v5_attempt(request, batch=batch, scene_root=scene_root)
    room = str(request["room_type"])
    source_seed = int(request["source_scene_seed"])
    formal_seed = int(request["formal_episode_seed"])
    corridor_root = (
        scene_root
        / f"{room}_seed{source_seed}"
        / "kinofail_corridor_v2_tracking030"
    )
    formal_records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((corridor_root / "o4_pairs").glob("*/pair_manifest.json")):
        value = _json(path)
        if value.get("protocol_role") == "formal":
            formal_records.append((path, value))
    base["checks"]["exactly_one_or_zero_total_formal_runs"] = len(formal_records) <= 1

    # Rejections before formal collection are already terminal and fully audited by v5.
    if not formal_records:
        base["integrity_passed"] = (
            base["terminal"] and bool(base["checks"]) and all(base["checks"].values())
        )
        return base

    formal_path, formal = formal_records[0]
    v6_path = formal_path.parent / "phase_aware_v6_formal_audit.json"
    independence_path = formal_path.parent / "formal_admission_audit.json"
    v6 = _json(v6_path) if v6_path.is_file() else None
    independence = _json(independence_path) if independence_path.is_file() else None
    expected_thresholds = batch["phase_aware_visual_contract"]["thresholds"]
    v6_checks: dict[str, bool] = {
        "single_formal_run_uses_preregistered_seed": len(formal_records) == 1
        and int(formal.get("seed", -1)) == formal_seed,
        "v6_audit_present": v6 is not None,
        "independence_audit_present": independence is not None,
    }
    if v6 is not None:
        inherited = v6.get("inherited_nonvisual_checks", {})
        audit_checks = v6.get("checks", {})
        input_spec = v6.get("input_pair_manifest", {})
        v6_checks.update(
            {
                "v6_schema": v6.get("schema_version")
                == "kinofail.embodiedgen-o4-pair-adjudication.v6",
                "v6_role_formal": v6.get("adjudication_role") == "formal",
                "v6_scene_id": v6.get("scene_id") == request["scene_id"],
                "v6_seed": int(v6.get("seed", -1)) == formal_seed,
                "v6_binds_pair_path": Path(input_spec.get("path", "")).resolve()
                == formal_path.resolve(),
                "v6_binds_pair_hash": input_spec.get("sha256") == _sha256(formal_path),
                "v6_artifacts_intact": bool(v6.get("artifact_checks"))
                and all(v6["artifact_checks"].values()),
                "v6_nonvisual_summary_consistent": bool(inherited)
                and audit_checks.get("all_nonvisual_pair_checks_passed")
                == all(inherited.values()),
                "v6_thresholds_frozen": v6.get("phase_aware_visuals", {}).get(
                    "thresholds"
                )
                == expected_thresholds,
                "v6_result_boolean": isinstance(v6.get("passed"), bool),
                "v6_result_consistent": v6.get("passed")
                == (bool(audit_checks) and all(audit_checks.values())),
            }
        )
    if independence is not None:
        formal_pairs = independence.get("formal_pairs", [])
        v6_checks.update(
            {
                "independence_passed": independence.get("passed") is True,
                "independence_exactly_this_pair": len(formal_pairs) == 1
                and Path(formal_pairs[0].get("manifest", "")).resolve()
                == formal_path.resolve(),
            }
        )

    base["checks"].update(v6_checks)
    base["stages"]["formal_v6_adjudication"] = {
        "path": str(v6_path),
        "sha256": _sha256(v6_path) if v6_path.is_file() else None,
        "passed": None if v6 is None else v6.get("passed") is True,
    }
    base["stages"]["formal_independence"] = {
        "path": str(independence_path),
        "sha256": _sha256(independence_path) if independence_path.is_file() else None,
        "passed": None if independence is None else independence.get("passed") is True,
    }
    if v6 is None or independence is None:
        base["outcome"] = "pending"
        base["terminal"] = False
    else:
        base["outcome"] = "formal_v6_passed" if v6.get("passed") is True else "formal_v6_failed"
        base["terminal"] = True
    base["integrity_passed"] = (
        base["terminal"] and bool(base["checks"]) and all(base["checks"].values())
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    batch_path = args.batch.resolve()
    batch = _json(batch_path)
    frozen_checks = {
        f"code::{name}": _hash_spec(spec) for name, spec in batch["frozen_code"].items()
    }
    development_checks = {
        f"development::{name}": _hash_spec(spec)
        for name, spec in batch["development_evidence"].items()
    }
    exclusion_checks = {
        f"exclusion::{index:02d}::{row['scene_id']}": _hash_spec(row)
        for index, row in enumerate(batch["calibration_exclusions"], start=1)
    }
    requests = batch["new_scene_requests"]
    request_checks = {
        "request_count": len(requests)
        == int(batch["inference_scope"]["requested_scene_instances"]),
        "unique_scene_ids": len({row["scene_id"] for row in requests}) == len(requests),
        "unique_scene_seeds": len({row["source_scene_seed"] for row in requests})
        == len(requests),
        "unique_episode_seeds": len({row["formal_episode_seed"] for row in requests})
        == len(requests),
        "source_and_episode_seeds_disjoint": not (
            {row["source_scene_seed"] for row in requests}
            & {row["formal_episode_seed"] for row in requests}
        ),
    }
    attempts = [
        _v6_attempt(request, batch=batch, scene_root=args.scene_root.resolve())
        for request in requests
    ]
    sealed = all(attempt["terminal"] for attempt in attempts)
    integrity = (
        all(frozen_checks.values())
        and all(development_checks.values())
        and all(exclusion_checks.values())
        and all(request_checks.values())
        and sealed
        and all(attempt["integrity_passed"] for attempt in attempts)
    )
    formal_passes = sum(attempt["outcome"] == "formal_v6_passed" for attempt in attempts)
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-phase-aware-v6-batch-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "requested_scene_instances": len(attempts),
        "formal_v6_passes": formal_passes,
        "primary_target_met": formal_passes
        >= int(batch["inference_scope"]["minimum_formal_passes_for_primary_target"]),
        "request_checks": request_checks,
        "frozen_input_checks": frozen_checks,
        "development_evidence_checks": development_checks,
        "calibration_exclusion_checks": exclusion_checks,
        "attempts": attempts,
        "interpretation": (
            "Every preregistered scene remains in the denominator. Only an independently audited "
            "formal pair passing the preregistered phase-aware v6 contract contributes a confirmation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
