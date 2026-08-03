#!/usr/bin/env python3
"""Audit a preregistered held-out-material O4 batch without exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_spec(spec: dict[str, Any]) -> bool:
    path = (ROOT / spec["path"]).resolve()
    return path.is_file() and _sha256(path) == spec["sha256"]


def _stage(path: Path, value: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
        "present": value is not None,
        "passed": None if value is None else value.get("passed") is True,
    }


def _attempt(request: dict[str, Any], *, scene_root: Path, seal: bool) -> dict[str, Any]:
    root = scene_root / request["output_directory"]
    cell = root / "route_surface_v7_confirmation" / request["material_id"]
    pair_root = cell / "o4_pairs" / request["formal_output_directory"]
    paths = {
        "source_manifest": root / "source_manifest.json",
        "source_preflight": root / "source_geometry_preflight_v1/source_geometry_preflight_audit.json",
        "base_compile": root / "kinofail_base_v1/compiled_scene_audit.json",
        "corridor_v2": root / "kinofail_corridor_v2_tracking030/compiled_scene_audit.json",
        "route_surface_v4": cell / "compiled_scene_audit.json",
        "rtx_v5": cell / "rtx_qa/rtx_scene_audit.json",
        "go2_v2": cell / "go2_qa/go2_scene_audit.json",
        "admission_v7": cell / "route_surface_v7_admission_audit.json",
        "protocol": cell / "formal_o4_protocol_route_surface_v7.json",
        "pair_manifest": pair_root / "pair_manifest.json",
        "phase_v6": pair_root / "phase_aware_v6_formal_audit.json",
        "independence_v6": pair_root / "formal_admission_audit.json",
    }
    values = {
        name: _json(path) if path.is_file() and path.suffix == ".json" else None
        for name, path in paths.items()
    }
    checks: dict[str, bool] = {}
    manifest = values["source_manifest"]
    if manifest is not None:
        checks.update(
            {
                "source_scene_id": manifest.get("scene_id") == request["scene_id"],
                "source_room_type": manifest.get("generation", {}).get("room_type")
                == request["room_type"],
                "source_seed": int(manifest.get("generation", {}).get("seed", -1))
                == int(request["source_scene_seed"]),
            }
        )
    preflight = values["source_preflight"]
    if preflight is not None:
        checks.update(
            {
                "preflight_schema": preflight.get("schema_version")
                == "kinofail.embodiedgen-source-geometry-preflight.v1",
                "preflight_scene_id": preflight.get("scene_id") == request["scene_id"],
            }
        )
    compiled = values["route_surface_v4"]
    if compiled is not None:
        material = compiled.get("appearance_contract", {}).get("material", {})
        checks.update(
            {
                "compiled_schema": compiled.get("schema_version")
                == "kinofail.embodiedgen-compiled-scene.v4-development",
                "compiled_material_id": material.get("id") == request["material_id"],
                "compiled_material_split": material.get("split")
                == request["material_split"],
            }
        )
    admission = values["admission_v7"]
    if admission is not None:
        checks.update(
            {
                "admission_schema": admission.get("schema_version")
                == "kinofail.embodiedgen-route-surface-v7-admission-audit.v1",
                "admission_material": admission.get("material", {}).get("id")
                == request["material_id"],
            }
        )
    protocol = values["protocol"]
    if protocol is not None:
        frozen_admission = protocol.get("frozen_files", {}).get(
            "route_surface_v7_admission", {}
        )
        checks.update(
            {
                "protocol_schema": protocol.get("schema_version")
                == "kinofail.embodiedgen-o4-formal-protocol.v3",
                "protocol_frozen": protocol.get("status") == "frozen",
                "protocol_seed": int(request["formal_episode_seed"])
                in [int(seed) for seed in protocol.get("formal_episode_seeds", [])],
                "protocol_binds_admission": paths["admission_v7"].is_file()
                and frozen_admission.get("sha256") == _sha256(paths["admission_v7"]),
            }
        )
    pair = values["pair_manifest"]
    if pair is not None:
        checks.update(
            {
                "pair_schema": pair.get("schema_version")
                == "kinofail.embodiedgen-o4-paired-sequence.v4",
                "pair_formal": pair.get("protocol_role") == "formal",
                "pair_scene_id": pair.get("scene_id") == request["scene_id"],
                "pair_seed": int(pair.get("seed", -1))
                == int(request["formal_episode_seed"]),
            }
        )
    phase = values["phase_v6"]
    if phase is not None:
        checks.update(
            {
                "phase_schema": phase.get("schema_version")
                == "kinofail.embodiedgen-o4-pair-adjudication.v6",
                "phase_formal": phase.get("adjudication_role") == "formal",
                "phase_seed": int(phase.get("seed", -1))
                == int(request["formal_episode_seed"]),
                "phase_binds_pair": paths["pair_manifest"].is_file()
                and phase.get("input_pair_manifest", {}).get("sha256")
                == _sha256(paths["pair_manifest"]),
            }
        )
    independence = values["independence_v6"]
    if independence is not None:
        formal_pairs = independence.get("formal_pairs", [])
        checks["independence_exact_pair"] = len(formal_pairs) == 1 and Path(
            formal_pairs[0].get("manifest", "")
        ).resolve() == paths["pair_manifest"].resolve()

    ordered = [
        "source_manifest",
        "source_preflight",
        "base_compile",
        "corridor_v2",
        "route_surface_v4",
        "rtx_v5",
        "go2_v2",
        "admission_v7",
        "protocol",
        "pair_manifest",
        "phase_v6",
        "independence_v6",
    ]
    outcome = "pending"
    terminal = False
    for name in ordered:
        value = values[name]
        if value is None:
            if seal:
                outcome = f"missing_{name}"
                terminal = True
            break
        if name == "protocol":
            continue
        if value.get("passed") is False:
            outcome = f"{name}_failed"
            terminal = True
            break
    else:
        outcome = (
            "formal_v7_passed"
            if values["phase_v6"].get("passed") is True
            and values["independence_v6"].get("passed") is True
            else "formal_v7_failed"
        )
        terminal = True
    expected_rejection = terminal and outcome.endswith("_failed")
    evidence_chain_complete = outcome in ("formal_v7_passed", "formal_v7_failed") or (
        expected_rejection and not outcome.startswith("missing_")
    )
    integrity = terminal and evidence_chain_complete and all(checks.values())
    return {
        "scene_id": request["scene_id"],
        "room_type": request["room_type"],
        "source_scene_seed": request["source_scene_seed"],
        "formal_episode_seed": request["formal_episode_seed"],
        "material_id": request["material_id"],
        "material_split": request["material_split"],
        "outcome": outcome,
        "terminal": terminal,
        "integrity_passed": integrity,
        "checks": checks,
        "stages": {
            name: _stage(paths[name], values[name]) for name in ordered
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    batch_path = args.batch.resolve()
    batch = _json(batch_path)
    frozen_checks = {
        f"code::{name}": _hash_spec(spec) for name, spec in batch["frozen_code"].items()
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
        _attempt(request, scene_root=args.scene_root.resolve(), seal=args.seal)
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
        and all(evidence_checks.values())
        and all(request_checks.values())
        and sealed
        and all(row["integrity_passed"] for row in attempts)
    )
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-route-surface-v7-batch-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
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
        "frozen_input_checks": frozen_checks,
        "development_evidence_checks": evidence_checks,
        "attempts": attempts,
        "interpretation": (
            "All preregistered requests remain in the denominator. Source or downstream gate "
            "rejections are reported as failures; only v6-native independent formal passes count."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite batch audit: {args.out}")
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not integrity:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
