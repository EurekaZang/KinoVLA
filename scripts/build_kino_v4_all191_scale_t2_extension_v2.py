#!/usr/bin/env python3
"""Build the formal all-191 Scale/T2 extension after metadata preflight.

V2 preserves every scientific field from the preregistered V1 breadth design
while supplying, before any V2 collection, the exact internal-collector hash
and record-derived per-scene allowed scope required by the runtime validator.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_kino_v4_all191_scale_t2_extension_v1 as v1  # noqa: E402


OUTPUT = ROOT / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v2"
DATA_ROOT = Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2")
DERIVED_ROOT = Path("/data/eureka/kinovla_outputs/kino_v4_all191_scale_t2_extension_v2")
FREEZE_DIR = ROOT / "outputs/freeze/kino_v4_all191_scale_provenance_v2"
AMENDMENT = FREEZE_DIR / "amendment_manifest.json"
PREDECESSOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
WRAPPER = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
V1_FREEZE = (
    ROOT
    / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v1/design/freeze_manifest.json"
)
V1_PREFLIGHT = Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v1/corpus")
BENCHMARK = "kinofail_kino_v4_all191_v2"
PROTOCOL = "kinofail-kino-v4-all191-scale-t2-extension-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def allowed(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    mapping = {
        "conditions": "condition",
        "counterfactual_group_ids": "counterfactual_group_id",
        "target_operators": "target_operator",
        "scene_families": "scene_family",
        "physical_realizations": "physical_realization",
        "geometry_profiles": "geometry_profile",
        "severity_ids": "severity_id",
    }
    result: dict[str, list[Any]] = {}
    for output_key, record_key in mapping.items():
        values = sorted({row[record_key] for row in rows if record_key in row}, key=str)
        if not values:
            raise RuntimeError(f"empty formal scope: {output_key}")
        result[output_key] = values
    return result


def collection_protocol(
    *, scene_id: str, schedule: Path, registry: Path, pairs: int
) -> dict[str, Any]:
    rows = jsonl(schedule)
    if len(rows) != pairs * 2:
        raise RuntimeError(f"incomplete Scale schedule: {scene_id}")
    return {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL}-{scene_id}-scale",
        "status": "frozen",
        "confirmatory": True,
        "benchmark_id": BENCHMARK,
        "scene_id": scene_id,
        "battery": "scale",
        "schedule_path": str(schedule),
        "schedule_sha256": sha256(schedule),
        "scene_registry_path": str(registry),
        "scene_registry_sha256": sha256(registry),
        "material_lock_path": str(v1.LOCK),
        "material_lock_sha256": sha256(v1.LOCK),
        "collector_path": str(PREDECESSOR),
        "collector_sha256": sha256(PREDECESSOR),
        "operational_wrapper_path": str(WRAPPER),
        "operational_wrapper_sha256": sha256(WRAPPER),
        "operational_amendment_path": str(AMENDMENT),
        "runtime_manifest_path": str(v1.RUNTIME_MANIFEST),
        "runtime_manifest_sha256": sha256(v1.RUNTIME_MANIFEST),
        "f0_manifest_path": str(v1.F0),
        "f0_manifest_sha256": sha256(v1.F0),
        "design_path": str(v1.DESIGN),
        "design_sha256": sha256(v1.DESIGN),
        "allowed": allowed(rows),
        "collection_contract": {
            "counterfactual_pairs": pairs,
            "physical_episodes": pairs * 2,
            "appearance_views_per_episode": 3,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "result_dependent_retry_permitted": False,
        },
    }


def normalized_scale(path: Path) -> list[dict[str, Any]]:
    keep = (
        "scene_id",
        "scene_index",
        "domain",
        "material_id",
        "material_slot",
        "condition",
        "target_operator",
        "active_operator",
        "attribution_category",
        "severity_id",
        "severity_rank",
        "parameter_interpolation",
        "physics_parameters",
        "physical_realization",
        "geometry_profile",
        "geometry_hash",
        "operator_seed",
        "physical_seed",
        "physical_nuisance",
        "operator_index",
        "replicate_index",
        "appearance_views",
    )
    return [{key: row[key] for key in keep} for row in jsonl(path)]


def main() -> int:
    for path in (PREDECESSOR, WRAPPER, V1_FREEZE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists() or FREEZE_DIR.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else FREEZE_DIR)

    invalid_summaries = list(V1_PREFLIGHT.glob("**/pair_summaries/*.json"))
    invalid_manifests = list(V1_PREFLIGHT.glob("**/scale/**/manifest.json"))
    issues = {
        issue
        for path in invalid_manifests
        for issue in load(path).get("runtime_validation", {}).get("issues", [])
    }
    if issues and issues != {"formal_protocol_allowed_scope_missing"}:
        raise RuntimeError(f"unexpected preflight issue set: {issues}")

    amendment = {
        "schema_version": "kinofail.reconfirmation-collector-provenance-amendment.v1",
        "status": "sealed_before_scale_or_t3_episode_acquisition",
        "passed": True,
        "sealed_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": BENCHMARK,
        "preflight_supersession": {
            "v1_design_freeze": str(V1_FREEZE),
            "v1_design_freeze_sha256": sha256(V1_FREEZE),
            "v1_corpus": str(V1_PREFLIGHT),
            "pair_summaries": len(invalid_summaries),
            "episode_manifests": len(invalid_manifests),
            "issue_set": sorted(issues),
            "v1_data_excluded_from_v2": True,
            "model_predictions_read": False,
            "method_endpoint_scores_read": False,
        },
        "correction": {
            "wrapper_path": str(WRAPPER),
            "wrapper_sha256": sha256(WRAPPER),
            "exact_hash_predecessor_path": str(PREDECESSOR),
            "exact_hash_predecessor_sha256": sha256(PREDECESSOR),
            "only_changes": [
                "bind the formal protocol to the wrapper's exact-hash predecessor",
                "add the schedule-derived per-scene allowed scope before V2 collection",
            ],
            "simulation_logic_changed": False,
            "sensor_or_feature_logic_changed": False,
            "threshold_or_analysis_changed": False,
        },
        "incident": {
            "affected_pair_ids": [],
            "authorized_reexecution_pair_ids": [],
        },
        "scientific_contract": {
            "v1_sampling_design_preserved_exactly": True,
            "seeds_and_physics_preserved_exactly": True,
            "result_dependent_retry_enabled": False,
            "v2_uses_a_new_corpus_root": True,
        },
        "source_sha256": {
            "wrapper": sha256(WRAPPER),
            "exact_hash_predecessor": sha256(PREDECESSOR),
            "builder": sha256(Path(__file__).resolve()),
        },
    }
    write(AMENDMENT, amendment)

    v1.OUTPUT = OUTPUT
    v1.DATA_ROOT = DATA_ROOT
    v1.DERIVED_ROOT = DERIVED_ROOT
    v1.BENCHMARK = BENCHMARK
    v1.PROTOCOL = PROTOCOL
    v1.__file__ = str(Path(__file__).resolve())
    v1._collection_protocol = collection_protocol
    result = int(v1.main())
    if result != 0:
        return result

    v1_scale = (
        ROOT
        / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v1/"
        "schedules/global/scale_schedule.jsonl"
    )
    v2_scale = OUTPUT / "schedules/global/scale_schedule.jsonl"
    identical_scientific_rows = normalized_scale(v1_scale) == normalized_scale(v2_scale)
    protocols = sorted(OUTPUT.glob("schedules/scenes/*/scale/collection_protocol.json"))
    scope_ok = all(
        len(load(path).get("allowed", {}).get("target_operators", [])) == 11
        and len(load(path).get("allowed", {}).get("counterfactual_group_ids", [])) == 22
        for path in protocols
    )
    supersession = {
        "schema_version": "kinofail.kino-v4-all191-v2-supersession-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": identical_scientific_rows and scope_ok and len(protocols) == 179,
        "v1_scientific_rows_preserved_exactly": identical_scientific_rows,
        "formal_protocol_count": len(protocols),
        "all_protocols_authorize_22_pairs_and_11_operators": scope_ok,
        "v2_collection_started_at_audit": False,
        "v1_preflight_data_in_v2": False,
        "source_sha256": {
            "v1_scale_schedule": sha256(v1_scale),
            "v2_scale_schedule": sha256(v2_scale),
            "amendment": sha256(AMENDMENT),
            "builder": sha256(Path(__file__).resolve()),
        },
    }
    write(OUTPUT / "design/supersession_audit.json", supersession)
    if supersession["passed"] is not True:
        raise RuntimeError("V2 supersession audit failed")
    print(json.dumps(supersession, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
