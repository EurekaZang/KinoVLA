#!/usr/bin/env python3
"""Final v7 batch audit with v6 adjudication priority and family exclusions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audit_embodiedgen_o4_route_surface_v7_batch import ROOT, _hash_spec, _json, _sha256
from audit_embodiedgen_o4_route_surface_v7_batch_v2 import _attempt_v2


def _cell(request: dict[str, Any], scene_root: Path) -> Path:
    return (
        scene_root
        / request["output_directory"]
        / "route_surface_v7_confirmation"
        / request["material_id"]
    )


def _attempt_v3(
    request: dict[str, Any],
    *,
    scene_root: Path,
    seal: bool,
    calibration_families: set[str],
) -> dict[str, Any]:
    base = _attempt_v2(request, scene_root=scene_root, seal=seal)
    cell = _cell(request, scene_root)
    pair_root = cell / "o4_pairs" / request["formal_output_directory"]
    pair_path = pair_root / "pair_manifest.json"
    phase_path = pair_root / "phase_aware_v6_formal_audit.json"
    independence_path = pair_root / "formal_admission_audit.json"
    failure_path = pair_root / "independence_failure_audit.json"
    pair = _json(pair_path) if pair_path.is_file() else None
    phase = _json(phase_path) if phase_path.is_file() else None
    independence = _json(independence_path) if independence_path.is_file() else None
    failure = _json(failure_path) if failure_path.is_file() else None
    base["stages"]["independence_exception"] = {
        "path": str(failure_path),
        "sha256": _sha256(failure_path) if failure_path.is_file() else None,
        "present": failure is not None,
        "passed": None if failure is None else failure.get("passed") is True,
    }

    # The registered v6 adjudication replaces the collector's stale all-frame visual aggregate.
    # A raw pair false therefore cannot terminate an otherwise complete v6 evidence chain.
    if pair is not None and phase is not None and independence is not None:
        base["checks"]["v6_adjudication_has_priority_over_raw_pair"] = True
        base["outcome"] = (
            "formal_v7_passed"
            if phase.get("passed") is True and independence.get("passed") is True
            else "formal_v7_failed"
        )
        base["terminal"] = True
        base["integrity_passed"] = all(base["checks"].values())
        return base

    if failure is None:
        return base

    required = {
        Path(row.get("path", "")).resolve(): row.get("sha256")
        for row in failure.get("required_inputs", [])
    }
    command_files = failure.get("command_files", [])
    expected_inputs = [pair_path, phase_path, cell / "formal_o4_protocol_route_surface_v7.json"]
    scene_family = None if pair is None else str(pair.get("scene_family"))
    failure_checks = {
        "independence_exception_schema": failure.get("schema_version")
        == "kinofail.frozen-stage-execution-audit.v1",
        "independence_exception_stage": failure.get("stage")
        == "v6_native_independence",
        "independence_exception_scene": failure.get("scene_id") == request["scene_id"],
        "independence_exception_nonzero_exit": int(failure.get("exit_code", 0)) != 0,
        "independence_exception_marked_failed": failure.get("passed") is False,
        "independence_exception_binds_inputs": all(
            path.is_file()
            and required.get(path.resolve()) == _sha256(path.resolve())
            for path in expected_inputs
        ),
        "independence_exception_binds_frozen_auditor": any(
            Path(row.get("path", "")).resolve()
            == (ROOT / "scripts/audit_embodiedgen_o4_independence_v6.py").resolve()
            and row.get("sha256")
            == _sha256(ROOT / "scripts/audit_embodiedgen_o4_independence_v6.py")
            for row in command_files
        ),
        "independence_native_audit_absent": independence is None,
        "calibration_family_overlap_confirmed": scene_family in calibration_families,
        "frozen_protocol_violation_reported": "violates frozen protocol"
        in str(failure.get("stderr", "")),
    }
    base["checks"].update(failure_checks)
    base["outcome"] = "independence_v6_failed"
    base["terminal"] = True
    base["integrity_passed"] = all(base["checks"].values())
    base["administrative_interpretation"] = (
        "The formal scene reused a calibration scene family and was rejected by the frozen "
        "v6 independence contract. It remains a full-denominator failure without replacement."
    )
    return base


def _amendment_checks(
    amendment: dict[str, Any], *, batch_path: Path, expected_status: str
) -> dict[str, bool]:
    return {
        "schema": amendment.get("schema_version")
        == "kinofail.embodiedgen-o4-route-surface-v7-audit-amendment.v1",
        "status": amendment.get("status") == expected_status,
        "batch_path": Path(amendment.get("batch", {}).get("path", "")).resolve()
        == batch_path,
        "batch_hash": amendment.get("batch", {}).get("sha256") == _sha256(batch_path),
        "no_scientific_contract_change": amendment.get("scientific_contract_changed") is False,
        "no_retry_or_replacement": amendment.get("retry_or_replacement_authorized") is False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--amendment1", type=Path, required=True)
    parser.add_argument("--amendment2", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    batch_path = args.batch.resolve()
    amendment1_path = args.amendment1.resolve()
    amendment2_path = args.amendment2.resolve()
    batch = _json(batch_path)
    amendment1 = _json(amendment1_path)
    amendment2 = _json(amendment2_path)
    frozen_checks = {
        f"code::{name}": _hash_spec(spec) for name, spec in batch["frozen_code"].items()
    }
    addition_checks = {
        **{
            f"amendment1::{name}": _hash_spec(spec)
            for name, spec in amendment1["frozen_additions"].items()
        },
        **{
            f"amendment2::{name}": _hash_spec(spec)
            for name, spec in amendment2["frozen_additions"].items()
        },
    }
    amendment_checks = {
        **{
            f"amendment1::{name}": value
            for name, value in _amendment_checks(
                amendment1,
                batch_path=batch_path,
                expected_status="frozen_before_corridor_refinement_or_heldout_render",
            ).items()
        },
        **{
            f"amendment2::{name}": value
            for name, value in _amendment_checks(
                amendment2,
                batch_path=batch_path,
                expected_status="frozen_after_collection_before_final_batch_seal",
            ).items()
        },
        "amendment2_binds_amendment1": amendment2.get("prior_amendment", {}).get(
            "sha256"
        )
        == _sha256(amendment1_path),
    }
    evidence_checks = {
        f"development::{name}": _hash_spec(spec)
        for name, spec in batch["development_evidence"].items()
    }
    calibration_paths = [
        (ROOT / row["path"]).resolve() for row in batch["calibration_exclusions"]
    ]
    calibration_families = {
        str(_json(path)["scene_family"]) for path in calibration_paths
    }
    calibration_checks = {
        f"calibration::{index:02d}": path.is_file()
        and _sha256(path) == row["sha256"]
        for index, (path, row) in enumerate(
            zip(calibration_paths, batch["calibration_exclusions"], strict=True), start=1
        )
    }
    requests = batch["new_scene_requests"]
    request_checks = {
        "request_count": len(requests)
        == int(batch["inference_scope"]["requested_scene_instances"]),
        "unique_scene_ids": len({row["scene_id"] for row in requests}) == len(requests),
        "unique_source_seeds": len({row["source_scene_seed"] for row in requests})
        == len(requests),
        "unique_episode_seeds": len({row["formal_episode_seed"] for row in requests})
        == len(requests),
        "source_episode_seeds_disjoint": not (
            {row["source_scene_seed"] for row in requests}
            & {row["formal_episode_seed"] for row in requests}
        ),
        "all_materials_unique": len({row["material_id"] for row in requests})
        == len(requests),
        "balanced_val_test": sum(row["material_split"] == "val" for row in requests)
        == sum(row["material_split"] == "test" for row in requests),
    }
    attempts = [
        _attempt_v3(
            request,
            scene_root=args.scene_root.resolve(),
            seal=args.seal,
            calibration_families=calibration_families,
        )
        for request in requests
    ]
    sealed = args.seal and all(row["terminal"] for row in attempts)
    formal_passes = [row for row in attempts if row["outcome"] == "formal_v7_passed"]
    passes_by_split = {
        split: sum(row["material_split"] == split for row in formal_passes)
        for split in ("val", "test")
    }
    passed_families = len({row["room_type"] for row in formal_passes})
    passed_materials = len({row["material_id"] for row in formal_passes})
    target = batch["inference_scope"]
    target_checks = {
        "total_formal_passes": len(formal_passes)
        >= int(target["minimum_formal_passes_for_primary_target"]),
        "val_formal_passes": passes_by_split["val"]
        >= int(target["minimum_passes_per_material_split"]),
        "test_formal_passes": passes_by_split["test"]
        >= int(target["minimum_passes_per_material_split"]),
        "room_family_coverage": passed_families
        >= int(target["minimum_room_families_among_passes"]),
        "material_coverage": passed_materials
        >= int(target["minimum_distinct_materials_among_passes"]),
    }
    integrity = (
        all(frozen_checks.values())
        and all(addition_checks.values())
        and all(amendment_checks.values())
        and all(evidence_checks.values())
        and all(calibration_checks.values())
        and all(request_checks.values())
        and sealed
        and all(row["integrity_passed"] for row in attempts)
    )
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-route-surface-v7-batch-audit.v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "amendments": [
            {"path": str(amendment1_path), "sha256": _sha256(amendment1_path)},
            {"path": str(amendment2_path), "sha256": _sha256(amendment2_path)},
        ],
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "requested_scene_instances": len(attempts),
        "formal_v7_passes": len(formal_passes),
        "passes_by_material_split": passes_by_split,
        "passed_room_families": passed_families,
        "passed_distinct_materials": passed_materials,
        "calibration_scene_families": sorted(calibration_families),
        "primary_target_checks": target_checks,
        "primary_target_met": all(target_checks.values()),
        "request_checks": request_checks,
        "amendment_checks": amendment_checks,
        "calibration_checks": calibration_checks,
        "frozen_input_checks": frozen_checks,
        "frozen_addition_checks": addition_checks,
        "development_evidence_checks": evidence_checks,
        "attempts": attempts,
        "interpretation": (
            "Every preregistered request remains in the denominator. Phase-aware v6 is the "
            "registered terminal adjudicator; family overlap with calibration is an independence "
            "failure. Amendments only preserve exceptions and aggregation semantics."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite batch audit: {args.out}")
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 2 if sealed and not integrity else 0


if __name__ == "__main__":
    raise SystemExit(main())
