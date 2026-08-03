#!/usr/bin/env python3
"""Seal the realized v2 scenes, assets, schedules, and attrition before data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus",
    ROOT / "outputs/kinofail_reconfirmation_v2/c2_t2",
    ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": _display(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_f0(path: Path) -> dict[str, Any]:
    manifest = _json(path)
    sidecar = path.with_name("freeze_manifest.sha256")
    if (
        not sidecar.is_file()
        or _sha256(path) != sidecar.read_text(encoding="utf-8").split()[0]
        or manifest.get("status") != "frozen_before_new_scene_generation"
        or manifest.get("confirmatory") is not True
        or manifest.get("new_data_available_at_freeze") is not False
        or manifest.get("model_predictions_available_at_freeze") is not False
    ):
        raise RuntimeError("invalid reconfirmation F0/amendment")
    for row in manifest.get("frozen_files", []):
        frozen = _resolve(str(row["path"]))
        if (
            not frozen.is_file()
            or frozen.stat().st_size != int(row["bytes"])
            or _sha256(frozen) != str(row["sha256"])
        ):
            raise RuntimeError(f"F0-frozen file changed: {frozen}")
    return manifest


def _scene_evidence(registry: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for row in registry["scenes"]:
        for key in (
            "episode_usd",
            "compiled_audit",
            "terminal_scene_admission",
        ):
            paths.append(_resolve(str(row[key])))
        contract = row.get("geometry_hash_contract", {})
        for source in contract.get("sources", []):
            paths.append(_resolve(str(source["path"])))
        if contract.get("source"):
            paths.append(_resolve(str(contract["source"])))
    for domain in ("life", "production", "wild"):
        for attempt in registry["attrition"][domain]["attempts"]:
            paths.append(_resolve(str(attempt["terminal_audit"])))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--hdri-lock", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--freshness-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or any(path.exists() for path in FORBIDDEN):
        raise FileExistsError(
            "F1 must precede all anomaly, conflict, feature, and prediction data"
        )

    f0_path = args.f0_manifest.resolve()
    registry_path = args.scene_registry.resolve()
    material_path = args.material_lock.resolve()
    hdri_path = args.hdri_lock.resolve()
    schedules = args.schedule_root.resolve()
    freshness_path = args.freshness_audit.resolve()
    f0 = _verify_f0(f0_path)
    registry = _json(registry_path)
    material = _json(material_path)
    hdri = _json(hdri_path)
    freshness = _json(freshness_path)
    schedule_manifest_path = schedules / "manifest.json"
    schedule_manifest = _json(schedule_manifest_path)

    scenes = [dict(row) for row in registry.get("scenes", [])]
    scene_sources = {str(row["source_scene_id"]) for row in scenes}
    origins = {
        "original_f0_unexposed": sum(
            str(row.get("origin", "")).startswith("original_f0_unexposed")
            for row in scenes
        ),
        "new_after_reconfirmation_f0": sum(
            row.get("origin") == "new_after_reconfirmation_f0"
            for row in scenes
        ),
    }
    attempts = sum(
        len(registry["attrition"][domain]["attempts"])
        for domain in ("life", "production", "wild")
    )
    admissions = sum(
        sum(
            attempt.get("admitted") is True
            for attempt in registry["attrition"][domain]["attempts"]
        )
        for domain in ("life", "production", "wild")
    )
    checks = {
        "f0_is_valid_and_unchanged": f0.get("confirmatory") is True,
        "scene_registry_has_30_unique_slots_and_sources": (
            registry.get("scene_count") == 30
            and len(scenes) == 30
            and len({str(row["scene_id"]) for row in scenes}) == 30
            and len(scene_sources) == 30
        ),
        "scene_selection_is_model_blind": (
            registry.get("selection_uses_model_predictions") is False
        ),
        "scene_origin_counts_exact": origins
        == {
            "original_f0_unexposed": 21,
            "new_after_reconfirmation_f0": 9,
        },
        "new_scene_attrition_below_frozen_ceiling": (
            attempts == 35
            and admissions == 30
            and (attempts - admissions) / 14 < 0.5
        ),
        "material_lock_passed_and_has_30": (
            material.get("audit", {}).get("passed") is True
            and material.get("audit", {}).get("n_materials") == 30
        ),
        "hdri_lock_passed": (
            hdri.get("passed") is True
            or hdri.get("audit", {}).get("passed") is True
        ),
        "freshness_audit_all_checks_pass": (
            freshness.get("passed") is True
            and all(freshness.get("checks", {}).values())
        ),
        "schedule_manifest_passed": schedule_manifest.get("passed") is True,
        "schedule_counts_exact": schedule_manifest.get("counts")
        == {
            "scene_shards": 30,
            "material_contexts": 60,
            "scale_counterfactual_pairs": 10_560,
            "scale_physical_episodes": 21_120,
            "t2_cases": 1_500,
            "t3_cases": 1_500,
            "t3_counterfactual_pairs": 3_000,
            "t3_physical_episodes": 6_000,
            "conflict_model_records": 18_000,
        },
        "schedule_binds_current_f0": (
            schedule_manifest.get("bindings", {}).get("f0_manifest_sha256")
            == _sha256(f0_path)
        ),
        "schedule_binds_current_registry_and_materials": (
            schedule_manifest.get("bindings", {}).get("scene_registry_sha256")
            == _sha256(registry_path)
            and schedule_manifest.get("bindings", {}).get(
                "material_lock_sha256"
            )
            == _sha256(material_path)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "F1 preconditions failed: "
            + json.dumps(checks, sort_keys=True)
        )

    paths = [
        f0_path,
        f0_path.with_name("freeze_manifest.sha256"),
        registry_path,
        material_path,
        hdri_path,
        freshness_path,
        Path(__file__).resolve(),
    ]
    paths.extend(_scene_evidence(registry))
    paths.extend(
        path for path in schedules.rglob("*") if path.is_file()
    )
    records = [
        _record(path)
        for path in sorted(set(path.resolve() for path in paths), key=str)
    ]
    manifest = {
        "schema_version": "kinofail.unified-reconfirmation-f1.v2",
        "protocol_id": (
            "kinofail-unified-moe-v3-independent-reconfirmation-f1-v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_anomaly_collection_and_model_inference",
        "confirmatory": True,
        "anomaly_data_available_at_seal": False,
        "model_predictions_available_at_seal": False,
        "model_or_endpoint_outcomes_used_for_scene_selection": False,
        "checks": checks,
        "counts": {
            "scenes": 30,
            "original_f0_unexposed_scenes": 21,
            "new_scenes": 9,
            "new_scene_attempts": 14,
            "new_scene_failures": 5,
            "materials": 30,
            "scale_counterfactual_pairs": 10_560,
            "t2_cases": 1_500,
            "t3_cases": 1_500,
        },
        "input_sha256": {
            "f0_manifest": _sha256(f0_path),
            "scene_registry": _sha256(registry_path),
            "material_lock": _sha256(material_path),
            "hdri_lock": _sha256(hdri_path),
            "schedule_manifest": _sha256(schedule_manifest_path),
            "freshness_audit": _sha256(freshness_path),
        },
        "frozen_realized_files": records,
        "forbidden_runtime_roots": [_display(path) for path in FORBIDDEN],
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "seal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "seal_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  seal_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "frozen_realized_files": len(records),
                "new_scene_attrition": "5/14",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
