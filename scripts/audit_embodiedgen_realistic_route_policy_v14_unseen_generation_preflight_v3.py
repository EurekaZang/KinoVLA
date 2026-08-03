#!/usr/bin/env python3
"""Final generation authorization after the two administrative amendments."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_generation_freeze_v1.json"
AMENDMENT1 = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_generation_freeze_v1_amendment1.json"
AMENDMENT2 = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_generation_freeze_v1_amendment2.json"
PREFLIGHT2 = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_generation_preflight_v2.json"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_generation_preflight_v3.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _bound(spec: dict[str, Any]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> int:
    config = _json(CONFIG)
    amendment1 = _json(AMENDMENT1)
    amendment2 = _json(AMENDMENT2)
    preflight2 = _json(PREFLIGHT2)
    request = config["new_untouched_scene_request"]
    command = amendment2["correction"]["effective_generation_command"]
    expected_command = [
        "/home/eureka/miniconda3/envs/kinovla/bin/python",
        "scripts/generate_embodiedgen_room_source.py",
        "--embodiedgen-root",
        "/home/eureka/dependencies/EmbodiedGen-v2.0.0",
        "--output-root",
        "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
        "--scene-id",
        "indoor_bathroom_33",
        "--room-type",
        "Bathroom",
        "--seed",
        "20261213",
        "--complexity",
        "simple",
    ]
    embodiedgen_root = Path("/home/eureka/dependencies/EmbodiedGen-v2.0.0")
    checks = {
        "prior_preflight_passed_and_bound": preflight2.get("passed") is True
        and _bound(amendment2["passed_preflight_v2_retained"]),
        "base_freeze_bound": _bound(amendment2["base_freeze"]),
        "amendment1_bound": _bound(amendment2["amendment1"]),
        "amendment1_supported": amendment1.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-freeze-amendment.v1",
        "amendment2_supported": amendment2.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-freeze-amendment.v1",
        "effective_command_exact": command == expected_command,
        "embodiedgen_checkout_present": (embodiedgen_root / ".git").exists(),
        "administrative_only": all(
            amendment2.get(key) is False
            for key in (
                "scientific_fields_changed",
                "scene_selection_changed",
                "seed_changed",
                "material_assignment_changed",
                "generator_child_command_changed",
                "runtime_or_threshold_changed",
            )
        ),
        "new_source_path_still_unused": not _resolve(request["source_output"]).exists(),
        "new_route_surface_path_still_unused": not _resolve(
            request["route_surface_output"]
        ).exists(),
        "new_runtime_path_still_unused": not _resolve(
            request["planned_runtime_output"]
        ).exists(),
    }
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-preflight.v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "effective_generation_command": command,
        "new_scene_generation_authorized": all(checks.values()),
        "runtime_collection_authorized": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "inputs": {
            "base_config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
            "amendment1": {"path": str(AMENDMENT1), "sha256": _sha256(AMENDMENT1)},
            "amendment2": {"path": str(AMENDMENT2), "sha256": _sha256(AMENDMENT2)},
            "preflight2": {"path": str(PREFLIGHT2), "sha256": _sha256(PREFLIGHT2)},
        },
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
