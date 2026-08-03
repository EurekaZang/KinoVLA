#!/usr/bin/env python3
"""Seal the development calibration of the route-motion visual proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (
            "indoor_livingroom_54",
            "LivingRoom_seed20261255/route_surface_o4_v23_stratified_confirmation/"
            "test_concrete039",
        ),
        (
            "indoor_livingroom_55",
            "LivingRoom_seed20261257/route_surface_o4_v23_stratified_confirmation/"
            "val_ground071",
        ),
        (
            "indoor_office_58",
            "Office_seed20261263/route_surface_o4_v23_stratified_confirmation/"
            "val_tiles139",
        ),
        (
            "indoor_diningroom_59",
            "DiningRoom_seed20261265/route_surface_o4_v23_stratified_confirmation/"
            "test_ground073",
        ),
    ]
    source_root = ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2"
    rows = []
    for scene_id, relative in cases:
        route = source_root / relative
        proxy_path = route / "motion_rtx_proxy_v2_extrinsics/route_motion_rtx_audit.json"
        go2_path = route / "terrain_route_v2/go2_qa/go2_scene_audit.json"
        proxy = _json(proxy_path)
        go2 = _json(go2_path)
        proxy_passed = proxy.get("passed") is True
        go2_visual_passed = go2.get("checks", {}).get("front_frames_non_degenerate") is True
        rows.append(
            {
                "scene_id": scene_id,
                "proxy": {"path": str(proxy_path), "sha256": _sha256(proxy_path)},
                "articulated_go2": {"path": str(go2_path), "sha256": _sha256(go2_path)},
                "proxy_passed": proxy_passed,
                "articulated_front_views_passed": go2_visual_passed,
                "agreement": proxy_passed == go2_visual_passed,
            }
        )
    checks = {
        "four_fixed_v23_cases": len(rows) == 4,
        "contains_both_classes": {row["proxy_passed"] for row in rows} == {False, True},
        "four_of_four_agreement": all(row["agreement"] for row in rows),
        "proxy_is_operator_blind": all(
            _json(Path(row["proxy"]["path"])).get("operator_blind") is True for row in rows
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.embodiedgen-route-motion-proxy-calibration.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "sealed": True,
        "checks": checks,
        "cases": rows,
        "interpretation": (
            "Development calibration only: the extrinsics-corrected proxy agrees with the "
            "existing articulated Go2 front-view gate on all four eligible v23 scenes."
        ),
        "counts_as_realistic_corpus_evidence": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
