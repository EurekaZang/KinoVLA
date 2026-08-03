#!/usr/bin/env python3
"""Seal the deterministic Bathroom33 scene-build rejection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.sim.embodiedgen_scene import compile_embodiedgen_kinofail_scene


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_generation_preflight_v3.json"
SCENE_ROOT = ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/Bathroom_seed20261213"
SOURCE_MANIFEST = SCENE_ROOT / "source_manifest.json"
SOURCE_INTEGRITY = SCENE_ROOT / "source_integrity_audit.json"
SOURCE_PREFLIGHT = SCENE_ROOT / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"
BASE_OUTPUT = SCENE_ROOT / "kinofail_base_v1"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_scene_build_postrun_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    preflight = _json(PREFLIGHT)
    manifest = _json(SOURCE_MANIFEST)
    source_integrity = _json(SOURCE_INTEGRITY)
    source_preflight = _json(SOURCE_PREFLIGHT)
    exception_type = None
    exception_message = None
    try:
        compile_embodiedgen_kinofail_scene(SOURCE_MANIFEST, BASE_OUTPUT)
    except Exception as exc:  # The rejection class/message are part of the sealed evidence.
        exception_type = type(exc).__name__
        exception_message = str(exc)
    checks = {
        "generation_preflight_passed": preflight.get("passed") is True,
        "source_identity": manifest.get("scene_id") == "indoor_bathroom_33"
        and manifest.get("generation", {}).get("room_type") == "Bathroom"
        and int(manifest.get("generation", {}).get("seed", -1)) == 20261213,
        "source_integrity_passed": source_integrity.get("passed") is True
        and source_integrity.get("scene_id") == "indoor_bathroom_33",
        "source_geometry_passed": source_preflight.get("passed") is True,
        "deterministic_base_compile_rejected": exception_type == "ValueError"
        and exception_message == "no collision-free route endpoint",
        "no_compiled_audit": not (BASE_OUTPUT / "compiled_scene_audit.json").exists(),
        "no_compiled_episode": not (BASE_OUTPUT / "episode.usda").exists(),
        "no_route_surface_stage": not (SCENE_ROOT / "route_surface_v7_unseen_confirmation").exists(),
        "no_runtime_collection": not (
            ROOT
            / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_confirmation_v1/bathroom33"
        ).exists(),
    }
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-unseen-scene-build-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "checks": checks,
        "scene_id": "indoor_bathroom_33",
        "sampling_outcome": "base_compile_rejected_no_collision_free_route_endpoint",
        "counts_in_requested_scene_denominator": True,
        "replacement_seed_authorized": False,
        "runtime_collection_authorized": False,
        "exception": {"type": exception_type, "message": exception_message},
        "evidence": {
            "generation_preflight": {"path": str(PREFLIGHT), "sha256": _sha256(PREFLIGHT)},
            "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": _sha256(SOURCE_MANIFEST)},
            "source_integrity": {"path": str(SOURCE_INTEGRITY), "sha256": _sha256(SOURCE_INTEGRITY)},
            "source_preflight": {"path": str(SOURCE_PREFLIGHT), "sha256": _sha256(SOURCE_PREFLIGHT)},
        },
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "next_required_action": (
            "Preregister a multi-scene extension in which every requested scene remains in "
            "the admission denominator; run the unchanged nominal policy on every admitted scene."
        ),
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "audit_passed": result["audit_passed"],
                "sampling_outcome": result["sampling_outcome"],
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
