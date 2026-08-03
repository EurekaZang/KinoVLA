#!/usr/bin/env python3
"""Freeze new scene/material streams and physical design before generation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
METHOD_F0 = ROOT / "outputs/freeze/kino_v4_dinov2large_terrain448_f0/freeze_manifest.json"
EXPECTED_METHOD_F0_SHA256 = (
    "c149f23a1de34feb723ca0a6d6cd7cd815f0eb1198eb8da8d11341a9a800dc87"
)
DESIGN = ROOT / "configs/data/kinofail_kino_v4_confirmation_design_v1.json"
SCENES = ROOT / "configs/data/kinofail_kino_v4_confirmation_scene_candidates_v1.json"
MATERIALS = ROOT / "configs/data/kinofail_kino_v4_confirmation_terrain_pbr_v1.yaml"
DATA_ROOT = Path("/data/eureka/kinofail_kino_v4_confirmation_v1")
OUTPUT = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1"


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


def _prior_material_ids() -> set[str]:
    result: set[str] = set()
    for path in (ROOT / "configs/data").glob("*.yaml"):
        if path.resolve() == MATERIALS.resolve():
            continue
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            continue
        for row in value.get("materials", []):
            if isinstance(row, dict) and row.get("source_asset_id"):
                result.add(str(row["source_asset_id"]))
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if DATA_ROOT.exists():
        raise RuntimeError("confirmation data root exists before F1")
    if not METHOD_F0.is_file() or _sha256(METHOD_F0) != EXPECTED_METHOD_F0_SHA256:
        raise RuntimeError("method F0 authority drift")
    method = _json(METHOD_F0)
    if method.get("status") != "frozen_before_independent_confirmation_generation":
        raise RuntimeError("method is not frozen for confirmation")
    design = _json(DESIGN)
    scenes = _json(SCENES)
    materials = yaml.safe_load(MATERIALS.read_text(encoding="utf-8"))
    rows = [dict(row) for row in materials["materials"]]
    ids = {str(row["id"]) for row in rows}
    sources = {str(row["source_asset_id"]) for row in rows}
    if len(rows) != 12 or len(ids) != 12 or len(sources) != 12:
        raise RuntimeError("confirmation material catalog must have 12 unique rows")
    overlap = sources & _prior_material_ids()
    if overlap:
        raise RuntimeError(f"confirmation materials overlap prior sources: {overlap}")
    by_domain = {
        domain: sum(domain in row.get("domains", []) for row in rows)
        for domain in ("life", "production", "wild")
    }
    if by_domain != {"life": 4, "production": 4, "wild": 4}:
        raise RuntimeError(f"material domain count drift: {by_domain}")
    indoor = [dict(row) for row in scenes["indoor_candidates"]]
    wild = [dict(row) for row in scenes["wild_scenes"]]
    if (
        sum(row["domain"] == "life" for row in indoor) < 4
        or sum(row["domain"] == "production" for row in indoor) < 4
        or len(wild) < 4
    ):
        raise RuntimeError("candidate streams cannot realize four scenes per domain")
    candidate_ids = [str(row["scene_id"]) for row in [*indoor, *wild]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("duplicate confirmation candidate ID")
    old_registry = _json(ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json")
    old_sources = {
        str(row["source_scene_id"]) for row in old_registry.get("scenes", [])
    }
    if set(candidate_ids) & old_sources:
        raise RuntimeError("confirmation scene sources overlap method-development scenes")
    moderate = set(design["scale"]["moderate_lambda_points"])
    hard = set(design["scale"]["hard_lambda_points"])
    old_points = {
        0.2025, 0.2475, 0.2925, 0.3375, 0.3825, 0.4275, 0.4725,
        0.5175, 0.73625, 0.76875, 0.80125, 0.83375, 0.86625,
        0.89875, 0.93125, 0.96375,
    }
    if moderate & old_points or hard & old_points or moderate & hard:
        raise RuntimeError("confirmation operator parameter points are not new")
    if design["scale"]["counterfactual_pairs"] != 2112:
        raise RuntimeError("Scale count contract drift")
    if design["conflict"]["cases_total"] != 576:
        raise RuntimeError("Conflict count contract drift")

    frozen_paths = [
        DESIGN,
        SCENES,
        MATERIALS,
        ROOT / "scripts/freeze_kinofail_kino_v4_confirmation_f1.py",
        ROOT / "scripts/run_kinofail_kino_v4_confirmation_indoor_scene.py",
        ROOT / "scripts/run_kinofail_kino_v4_confirmation_wild_scene.py",
        ROOT / "scripts/run_kinofail_confirmatory_indoor_scene_v1.py",
        ROOT / "scripts/run_kinofail_confirmatory_wild_scene_v1.py",
        ROOT / "scripts/sync_terrain_pbr_assets.py",
        METHOD_F0,
    ]
    manifest = {
        "schema_version": "kinofail.kino-v4-confirmation-freeze-f1.v1",
        "status": "frozen_before_material_sync_and_scene_generation",
        "created_utc": datetime.now(UTC).isoformat(),
        "confirmatory": True,
        "new_data_available_at_freeze": False,
        "method_f0": {
            "path": str(METHOD_F0.relative_to(ROOT)),
            "sha256": _sha256(METHOD_F0),
        },
        "design": design,
        "material_source_ids": sorted(sources),
        "prior_material_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(_prior_material_ids())).encode("utf-8")
        ).hexdigest(),
        "scene_candidate_ids": candidate_ids,
        "selection_uses_model_predictions": False,
        "data_root": str(DATA_ROOT),
        "frozen_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in frozen_paths
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    path = OUTPUT / "freeze_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "manifest": str(path.relative_to(ROOT)),
        "manifest_sha256": _sha256(path),
        "status": manifest["status"],
        "new_data_available_at_freeze": False,
    }
    (OUTPUT / "FREEZE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
