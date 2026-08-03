#!/usr/bin/env python3
"""Freeze the 18-pair realistic A7 legacy-surrogate collection before model outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/eval/kinofail_realistic_a7_matched_design_v1.json"
BASE_SCHEDULE = (
    ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_a7_legacy_surrogate_v1.py"
RUNTIME_MANIFEST = ROOT / "kino_vla/data/runtime_manifest.py"
OUT_DIR = ROOT / "outputs/kinofail_realistic/design_a7_legacy_surrogate_v1"
OUT_SCHEDULE = OUT_DIR / "schedule.jsonl"
OUT_PROTOCOL = ROOT / "configs/data/kinofail_realistic_a7_legacy_surrogate_formal_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _surrogate_parameters(row: dict[str, Any]) -> dict[str, float]:
    source = dict(row["physics_parameters"])
    if row["target_operator"] == "O2_compliance":
        return {
            "stiffness_n_per_m": float(source["stiffness_n_per_m"]),
            "damping_ns_per_m": float(source["shear_gain"]),
            "sink_depth_m": float(source["sink_depth_m"]),
        }
    if row["target_operator"] == "O4_tether":
        return {
            "stiffness_n_per_m": 120.0,
            "damping_ns_per_m": 2.0,
            "free_length_m": 0.0,
            "snap_force_n": 1.0e9,
            "peel_factor": 0.3,
            "force_cap_n": float(source["tangential_force_cap_n"]),
        }
    raise ValueError(row["target_operator"])


def main() -> int:
    if OUT_SCHEDULE.exists() or OUT_PROTOCOL.exists():
        raise FileExistsError(
            "A7 legacy-surrogate freeze already exists; refusing post-outcome overwrite"
        )
    design = _json(DESIGN)
    if design.get("model_outcomes_available_at_freeze") is not False:
        raise RuntimeError("A7 design is not certified outcome-blind")
    expected_base_hash = str(design["base_schedule_sha256"])
    if _sha256(BASE_SCHEDULE) != expected_base_hash:
        raise RuntimeError("A7 base schedule hash mismatch")
    selected_ids = set(
        design["design"]["local_vs_surrogate_physics"]["counterfactual_group_ids"]
    )
    rows = [
        row for row in _jsonl(BASE_SCHEDULE)
        if row["counterfactual_group_id"] in selected_ids
    ]
    if len(rows) != 2 * len(selected_ids):
        raise RuntimeError("A7 legacy-surrogate selection is not 18 complete pairs")

    frozen_rows = []
    for source in rows:
        row = dict(source)
        row["benchmark_id"] = "kinofail_realistic_a7_legacy_surrogate_v1"
        row["physical_realization"] = "legacy_base_wrench_surrogate"
        row["physical_nuisance_profile_index"] = 2
        row["physics_parameters"] = _surrogate_parameters(source)
        row["a7_ablation"] = {
            "cell": "surrogate_trunk_effect_vs_local_physics",
            "arm": "legacy_base_wrench_surrogate",
            "base_benchmark_id": source["benchmark_id"],
            "base_physical_realization": source["physical_realization"],
            "base_physics_parameters": source["physics_parameters"],
            "matched_base_counterfactual_group_id": source["counterfactual_group_id"],
        }
        frozen_rows.append(row)
    frozen_rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SCHEDULE.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen_rows),
        encoding="utf-8",
    )
    schedule_sha256 = _sha256(OUT_SCHEDULE)
    operators = sorted({str(row["target_operator"]) for row in frozen_rows})
    scenes = sorted({str(row["scene_family"]) for row in frozen_rows})
    realizations = sorted({str(row["physical_realization"]) for row in frozen_rows})
    geometries = sorted({str(row["geometry_profile"]) for row in frozen_rows})
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a7_legacy_surrogate_formal_v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_a7_legacy_surrogate_v1",
        "scope": (
            "36 episodes / 18 pairs / O2+O4 moderate / 9 scenes / nuisance profile 2; "
            "legacy base-wrench mechanism only; A8 excluded."
        ),
        "schedule_path": str(OUT_SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": schedule_sha256,
        "scene_registry_path": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": _sha256(REGISTRY),
        "collector_path": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": _sha256(COLLECTOR),
        "runtime_manifest_path": str(RUNTIME_MANIFEST.relative_to(ROOT)),
        "runtime_manifest_sha256": _sha256(RUNTIME_MANIFEST),
        "a7_design_path": str(DESIGN.relative_to(ROOT)),
        "a7_design_sha256": _sha256(DESIGN),
        "base_schedule_path": str(BASE_SCHEDULE.relative_to(ROOT)),
        "base_schedule_sha256": expected_base_hash,
        "allowed": {
            "counterfactual_group_ids": sorted(selected_ids),
            "target_operators": operators,
            "scene_families": scenes,
            "physical_realizations": realizations,
            "geometry_profiles": geometries,
            "conditions": ["anomaly", "nominal_counterfactual"],
            "severity_ids": ["moderate"],
        },
        "collection_contract": {
            "a8_in_scope": False,
            "counterfactual_pairs": len(selected_ids),
            "physical_episodes": len(frozen_rows),
            "scene_families": len(scenes),
            "physical_nuisance_profile_index": 2,
            "pair_shared_scene_camera_controller_and_nuisance": True,
            "only_mechanism_architecture_changes": True,
            "negative_or_null_results_retained": True,
            "model_outcomes_available_at_freeze": False,
        },
    }
    OUT_PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schedule": str(OUT_SCHEDULE.relative_to(ROOT)),
                "schedule_sha256": schedule_sha256,
                "protocol": str(OUT_PROTOCOL.relative_to(ROOT)),
                "protocol_sha256": _sha256(OUT_PROTOCOL),
                "pairs": len(selected_ids),
                "episodes": len(frozen_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
