#!/usr/bin/env python3
"""Audit route-surface v7 batches including recorded pre-audit stage failures."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audit_embodiedgen_o4_route_surface_v7_batch import (
    ROOT,
    _attempt as _v1_attempt,
    _hash_spec,
    _json,
    _sha256,
)


def _base_failure_path(request: dict[str, Any], scene_root: Path) -> Path:
    return (
        scene_root
        / request["output_directory"]
        / "kinofail_base_v1/base_compile_failure_audit.json"
    )


def _attempt_v2(
    request: dict[str, Any], *, scene_root: Path, seal: bool
) -> dict[str, Any]:
    base = _v1_attempt(request, scene_root=scene_root, seal=seal)
    failure_path = _base_failure_path(request, scene_root)
    failure = _json(failure_path) if failure_path.is_file() else None
    base["stages"]["base_compile_exception"] = {
        "path": str(failure_path),
        "sha256": _sha256(failure_path) if failure_path.is_file() else None,
        "present": failure is not None,
        "passed": None if failure is None else failure.get("passed") is True,
    }
    if failure is None:
        return base

    source_manifest = (
        scene_root / request["output_directory"] / "source_manifest.json"
    ).resolve()
    command_files = failure.get("command_files", [])
    required_inputs = failure.get("required_inputs", [])
    failure_checks = {
        "base_exception_schema": failure.get("schema_version")
        == "kinofail.frozen-stage-execution-audit.v1",
        "base_exception_stage": failure.get("stage") == "base_v1_compile",
        "base_exception_scene": failure.get("scene_id") == request["scene_id"],
        "base_exception_nonzero_exit": int(failure.get("exit_code", 0)) != 0,
        "base_exception_marked_failed": failure.get("passed") is False,
        "base_exception_binds_source": source_manifest.is_file()
        and len(required_inputs) == 1
        and Path(required_inputs[0].get("path", "")).resolve() == source_manifest
        and required_inputs[0].get("sha256") == _sha256(source_manifest),
        "base_exception_binds_frozen_compiler": any(
            Path(row.get("path", "")).resolve()
            == (ROOT / "scripts/compile_embodiedgen_kinofail_scene.py").resolve()
            and row.get("sha256")
            == _sha256(ROOT / "scripts/compile_embodiedgen_kinofail_scene.py")
            for row in command_files
        ),
        "base_native_audit_absent": not base["stages"]["base_compile"]["present"],
    }
    base["checks"].update(failure_checks)
    base["outcome"] = "base_compile_failed"
    base["terminal"] = True
    base["integrity_passed"] = all(base["checks"].values())
    base["administrative_interpretation"] = (
        "The frozen compiler exited on a deterministic route-planning exception before its "
        "native JSON audit was written; the exception is hash-bound and remains a denominator failure."
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    batch_path = args.batch.resolve()
    amendment_path = args.amendment.resolve()
    batch = _json(batch_path)
    amendment = _json(amendment_path)
    frozen_checks = {
        f"code::{name}": _hash_spec(spec) for name, spec in batch["frozen_code"].items()
    }
    addition_checks = {
        f"amendment::{name}": _hash_spec(spec)
        for name, spec in amendment["frozen_additions"].items()
    }
    amendment_checks = {
        "schema": amendment.get("schema_version")
        == "kinofail.embodiedgen-o4-route-surface-v7-audit-amendment.v1",
        "status": amendment.get("status")
        == "frozen_before_corridor_refinement_or_heldout_render",
        "batch_path": Path(amendment.get("batch", {}).get("path", "")).resolve()
        == batch_path,
        "batch_hash": amendment.get("batch", {}).get("sha256") == _sha256(batch_path),
        "no_scientific_contract_change": amendment.get("scientific_contract_changed") is False,
        "no_retry_or_replacement": amendment.get("retry_or_replacement_authorized") is False,
    }
    evidence_checks = {
        f"development::{name}": _hash_spec(spec)
        for name, spec in batch["development_evidence"].items()
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
        _attempt_v2(request, scene_root=args.scene_root.resolve(), seal=args.seal)
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
        and all(request_checks.values())
        and sealed
        and all(row["integrity_passed"] for row in attempts)
    )
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-route-surface-v7-batch-audit.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "amendment": {"path": str(amendment_path), "sha256": _sha256(amendment_path)},
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "requested_scene_instances": len(attempts),
        "formal_v7_passes": len(formal_passes),
        "passes_by_material_split": passes_by_split,
        "passed_room_families": passed_families,
        "passed_distinct_materials": passed_materials,
        "primary_target_checks": target_checks,
        "primary_target_met": all(target_checks.values()),
        "request_checks": request_checks,
        "amendment_checks": amendment_checks,
        "frozen_input_checks": frozen_checks,
        "frozen_addition_checks": addition_checks,
        "development_evidence_checks": evidence_checks,
        "attempts": attempts,
        "interpretation": (
            "Every preregistered request remains in the denominator. The amendment preserves "
            "deterministic compiler exceptions that occurred before native audit emission; it "
            "does not change any scene, material, threshold, or retry policy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite batch audit: {args.out}")
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 2 if sealed and not integrity else 0


if __name__ == "__main__":
    raise SystemExit(main())
