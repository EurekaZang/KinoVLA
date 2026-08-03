#!/usr/bin/env python3
"""Seal the realized model-blind confirmatory design before anomaly collection.

F0 freezes every choice that can be made before generating new scenes.  F1
binds the realized, model-blind scene prefix, downloaded asset locks, exact
per-scene schedules, and their freshness audit.  It refuses to run once any
confirmatory corpus, feature, prediction, truth-key, or score artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORBIDDEN = (
    ROOT / "outputs/kinofail_confirmatory_v1/corpus",
    ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1",
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
        raise TypeError(f"expected JSON object: {path}")
    return value


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_record(path: Path) -> dict[str, Any]:
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
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(path) != expected:
        raise RuntimeError("F0 manifest no longer matches its sidecar")
    if (
        manifest.get("status") != "frozen_before_new_scene_generation"
        or manifest.get("confirmatory") is not True
        or manifest.get("new_data_available_at_freeze") is not False
    ):
        raise RuntimeError("invalid F0 state")
    for row in manifest.get("frozen_files", []):
        frozen_path = Path(str(row["path"]))
        if not frozen_path.is_absolute():
            frozen_path = ROOT / frozen_path
        if (
            not frozen_path.is_file()
            or frozen_path.stat().st_size != int(row["bytes"])
            or _sha256(frozen_path) != str(row["sha256"])
        ):
            raise RuntimeError(f"F0-frozen file changed: {frozen_path}")
    return manifest


def _scene_evidence_paths(registry: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for row in registry.get("scenes", []):
        for key in (
            "episode_usd",
            "compiled_audit",
            "terminal_scene_admission",
        ):
            path = Path(str(row[key]))
            paths.append(path if path.is_absolute() else ROOT / path)
    for domain in ("life", "production", "wild"):
        for attempt in registry.get("attrition", {}).get(domain, {}).get("attempts", []):
            path = Path(str(attempt["terminal_audit"]))
            paths.append(path if path.is_absolute() else ROOT / path)
    return paths


def _recursive_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(root)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--hdri-lock", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--freshness-audit", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/freeze/unified_moe_v3_confirmatory_f1",
    )
    parser.add_argument("--forbid-existing", type=Path, action="append", default=[])
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite F1: {out}")
    forbidden = [path.resolve() for path in DEFAULT_FORBIDDEN]
    forbidden.extend(path.resolve() for path in args.forbid_existing)
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(
            "F1 must precede collection and inference; already exists: "
            + ", ".join(existing)
        )

    f0_path = args.f0_manifest.resolve()
    registry_path = args.scene_registry.resolve()
    material_path = args.material_lock.resolve()
    hdri_path = args.hdri_lock.resolve()
    schedule_root = args.schedule_root.resolve()
    freshness_path = args.freshness_audit.resolve()
    f0 = _verify_f0(f0_path)
    registry = _json(registry_path)
    material_lock = _json(material_path)
    hdri_lock = _json(hdri_path)
    freshness = _json(freshness_path)
    schedule_manifest_path = schedule_root / "manifest.json"
    schedule_manifest = _json(schedule_manifest_path)

    checks = {
        "f0_is_confirmatory": f0.get("confirmatory") is True,
        "scene_registry_has_30": (
            registry.get("scene_count") == 30
            and len(registry.get("scenes", [])) == 30
        ),
        "scene_selection_is_model_blind": (
            registry.get("selection_uses_model_predictions") is False
        ),
        "material_lock_passed": material_lock.get("audit", {}).get("passed") is True,
        "hdri_lock_passed": (
            hdri_lock.get("passed") is True
            or hdri_lock.get("audit", {}).get("passed") is True
        ),
        "freshness_audit_passed": (
            freshness.get("passed") is True and freshness.get("status") == "passed"
        ),
        "schedule_manifest_passed": schedule_manifest.get("passed") is True,
        "schedule_scale_count": (
            schedule_manifest.get("counts", {}).get("scale_counterfactual_pairs")
            == 10_560
        ),
        "schedule_conflict_counts": (
            schedule_manifest.get("counts", {}).get("t2_cases") == 1_500
            and schedule_manifest.get("counts", {}).get("t3_cases") == 1_500
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"F1 preconditions failed: {checks}")

    paths = [
        f0_path,
        f0_path.with_name("freeze_manifest.sha256"),
        registry_path,
        material_path,
        hdri_path,
        freshness_path,
        Path(__file__).resolve(),
    ]
    paths.extend(_scene_evidence_paths(registry))
    paths.extend(_recursive_files(schedule_root))
    records = [
        _file_record(path)
        for path in sorted({path.resolve() for path in paths}, key=str)
    ]
    manifest = {
        "schema_version": "kinofail.unified-confirmatory-f1.v1",
        "protocol_id": "kinofail-unified-moe-v3-confirmatory-f1-20260725",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_anomaly_collection_and_model_inference",
        "confirmatory": True,
        "anomaly_data_available_at_seal": False,
        "model_predictions_available_at_seal": False,
        "model_or_endpoint_outcomes_used_for_scene_selection": False,
        "checks": checks,
        "counts": {
            "scenes": 30,
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
        "forbidden_runtime_roots": [_display(path) for path in forbidden],
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
