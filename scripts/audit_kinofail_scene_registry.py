#!/usr/bin/env python3
"""Recompute scene-registry evidence and expose schedule/domain/operator gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kino_vla.sim.embodiedgen_asset import audit_embodiedgen_room_manifest


ROOT = Path(__file__).resolve().parents[1]
TARGET_DOMAINS = {"life", "production", "wild"}
TARGET_SPLITS = {"train", "val", "test"}
TARGET_OPERATORS = {f"O{index}" for index in range(1, 12)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_check(spec: dict[str, str]) -> tuple[Path, bool]:
    path = _resolve(spec["path"])
    return path, path.is_file() and _sha256(path) == spec["sha256"]


def _audit_scene(scene: dict[str, Any], *, verify_source_package: bool) -> dict[str, Any]:
    files: dict[str, Path] = {}
    checks: dict[str, bool] = {}
    for name in ("source_manifest", "compiled_audit", "rtx_audit", "go2_audit"):
        files[name], checks[f"{name}_hash"] = _file_check(scene[name])
    if not all(checks.values()):
        return {
            "scene_id": scene["scene_id"],
            "passed": False,
            "checks": checks,
            "issues": [key for key, passed in checks.items() if not passed],
        }

    source = _json(files["source_manifest"])
    compiled = _json(files["compiled_audit"])
    rtx = _json(files["rtx_audit"])
    go2 = _json(files["go2_audit"])
    episode_hash = str(rtx.get("episode_usd_sha256"))
    checks.update(
        {
            "source_scene_id": source.get("scene_id") == scene["scene_id"],
            "source_room_type": source.get("generation", {}).get("room_type")
            == scene["room_type"],
            "source_scene_seed": source.get("generation", {}).get("seed")
            == scene["source_scene_seed"],
            "compiled_passed": compiled.get("passed") is True,
            "compiled_scene_id": compiled.get("scene_id") == scene["scene_id"],
            "compiled_binds_source": compiled.get("source_manifest_sha256")
            == scene["source_manifest"]["sha256"],
            "rtx_passed": rtx.get("passed") is True,
            "rtx_binds_compiled": rtx.get("compiled_audit_sha256")
            == scene["compiled_audit"]["sha256"],
            "go2_passed": go2.get("passed") is True,
            "go2_scene_id": go2.get("scene_id") == scene["scene_id"],
            "go2_binds_compiled": go2.get("compiled_audit_sha256")
            == scene["compiled_audit"]["sha256"],
            "rtx_go2_same_episode": bool(episode_hash)
            and go2.get("episode_usd_sha256") == episode_hash,
            "all_rtx_checks_pass": all(bool(value) for value in rtx.get("checks", {}).values()),
            "all_go2_checks_pass": all(bool(value) for value in go2.get("checks", {}).values()),
            "known_domain": scene.get("domain") in TARGET_DOMAINS,
            "known_split": scene.get("split") in TARGET_SPLITS,
            "nonempty_operator_capability": bool(scene.get("operator_capabilities")),
        }
    )
    source_package_audit = None
    if verify_source_package:
        source_package_audit = audit_embodiedgen_room_manifest(files["source_manifest"])
        checks["source_package_integrity"] = source_package_audit.get("passed") is True

    return {
        "scene_id": scene["scene_id"],
        "scene_family": scene["scene_family"],
        "domain": scene["domain"],
        "split": scene["split"],
        "operator_capabilities": scene["operator_capabilities"],
        "passed": all(checks.values()),
        "checks": checks,
        "issues": [key for key, passed in checks.items() if not passed],
        "source_package_audit": source_package_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/data/kinofail_embodiedgen_scene_registry_v1.json",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=ROOT / "configs/data/kinofail_realistic.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_registry/registry_audit.json",
    )
    parser.add_argument(
        "--skip-source-package-hashes",
        action="store_true",
        help="Skip hashing every USD/texture file; registry evidence hashes are still checked.",
    )
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    registry = _json(registry_path)
    if registry.get("schema_version") != "kinofail.realistic-scene-registry.v1":
        raise ValueError("unsupported scene registry schema")
    scenes = list(registry.get("scenes", []))
    scene_ids = [str(scene["scene_id"]) for scene in scenes]
    scene_families = [str(scene["scene_family"]) for scene in scenes]
    source_seeds = [int(scene["source_scene_seed"]) for scene in scenes]
    uniqueness_checks = {
        "unique_scene_ids": len(scene_ids) == len(set(scene_ids)),
        "unique_scene_families": len(scene_families) == len(set(scene_families)),
        "unique_source_scene_seeds": len(source_seeds) == len(set(source_seeds)),
    }
    scene_audits = [
        _audit_scene(scene, verify_source_package=not args.skip_source_package_hashes)
        for scene in scenes
    ]

    config = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    scheduled_scenes = {
        str(scene["id"])
        for domain in config["domains"].values()
        for scene in domain["scenes"]
    }
    registered_scenes = set(scene_ids)
    validated_scenes = {
        str(audit["scene_id"]) for audit in scene_audits if audit["passed"]
    }
    capabilities = {
        str(operator)
        for scene in scenes
        if str(scene["scene_id"]) in validated_scenes
        for operator in scene["operator_capabilities"]
    }
    domain_counts = Counter(
        str(scene["domain"])
        for scene in scenes
        if str(scene["scene_id"]) in validated_scenes
    )
    split_counts = Counter(
        str(scene["split"])
        for scene in scenes
        if str(scene["scene_id"]) in validated_scenes
    )
    schedule_binding = {
        "scheduled_scene_count": len(scheduled_scenes),
        "registered_scheduled_scene_count": len(scheduled_scenes & registered_scenes),
        "validated_scheduled_scene_count": len(scheduled_scenes & validated_scenes),
        "missing_registry_entries": sorted(scheduled_scenes - registered_scenes),
        "registered_not_in_current_schedule": sorted(registered_scenes - scheduled_scenes),
        "fully_bound": scheduled_scenes <= validated_scenes,
    }
    coverage = {
        "validated_scene_count": len(validated_scenes),
        "domain_scene_counts": dict(sorted(domain_counts.items())),
        "split_scene_counts": dict(sorted(split_counts.items())),
        "operator_capabilities": sorted(capabilities),
        "missing_operator_capabilities": sorted(TARGET_OPERATORS - capabilities),
        "all_domains_have_three_scene_families": all(
            domain_counts[domain] >= 3 for domain in TARGET_DOMAINS
        ),
        "all_splits_present": all(split_counts[split] >= 1 for split in TARGET_SPLITS),
        "all_11_operators_scene_admitted": TARGET_OPERATORS <= capabilities,
    }
    registry_passed = all(uniqueness_checks.values()) and all(
        bool(audit["passed"]) for audit in scene_audits
    )
    publication_ready = (
        registry_passed
        and schedule_binding["fully_bound"]
        and coverage["all_domains_have_three_scene_families"]
        and coverage["all_splits_present"]
        and coverage["all_11_operators_scene_admitted"]
    )
    result = {
        "schema_version": "kinofail.realistic-scene-registry-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "registry": {
            "path": str(registry_path),
            "sha256": _sha256(registry_path),
            "registry_id": registry["registry_id"],
        },
        "passed": registry_passed,
        "publication_ready": publication_ready,
        "source_package_hashes_recomputed": not args.skip_source_package_hashes,
        "uniqueness_checks": uniqueness_checks,
        "scene_audits": scene_audits,
        "schedule_binding": schedule_binding,
        "coverage": coverage,
        "interpretation": (
            "Every listed registry entry is authentic and QA-valid, but publication readiness "
            "also requires the schedule, three domains, and all 11 operator capabilities."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not registry_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
