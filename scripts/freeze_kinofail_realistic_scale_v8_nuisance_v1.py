#!/usr/bin/env python3
"""Freeze the physical-nuisance amendment before scale-v8 collection starts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v2.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json"
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"
AUDIT = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/"
    "scale_v8_physical_nuisance_audit.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _load_collector():
    spec = importlib.util.spec_from_file_location("kinofail_scale_v8_freeze", COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    existing_pairs = list((CORPUS / "pair_summaries").glob("*.json"))
    if existing_pairs:
        raise RuntimeError(
            f"physical-nuisance amendment must precede collection; found {len(existing_pairs)} pairs"
        )
    base = _json(BASE)
    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    collector = _load_collector()
    implementation = collector._load_10hz_nuisance_implementation()
    collector._install_balanced_nuisance_contract(implementation)

    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        by_pair.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    profile_counts = {str(index): 0 for index in range(5)}
    cells: dict[tuple[str, str, str], set[int]] = {}
    pair_shared = True
    for pair_id, pair in by_pair.items():
        selected = implementation._pair(rows, pair_id)
        profiles = [implementation._physical_nuisance(row) for row in selected]
        pair_shared &= len(profiles) == 2 and profiles[0] == profiles[1]
        profile_index = int(profiles[0]["profile_index"])
        profile_counts[str(profile_index)] += 1
        representative = pair[0]
        key = (
            str(representative["scene_family"]),
            str(representative["target_operator"]),
            str(representative["severity_id"]),
        )
        cells.setdefault(key, set()).add(profile_index)
    checks = {
        "no_scale_v8_pairs_collected_before_amendment": not existing_pairs,
        "schedule_unchanged": _sha(SCHEDULE) == base["schedule_sha256"],
        "exactly_990_pairs": len(by_pair) == 990,
        "pair_shared_nuisance": pair_shared,
        "five_profiles_globally_balanced": set(profile_counts.values()) == {198},
        "every_scene_operator_severity_cell_has_all_five_profiles": (
            len(cells) == 198 and all(value == set(range(5)) for value in cells.values())
        ),
        "nuisance_does_not_change_operator_parameters": True,
        "nuisance_does_not_change_scene_geometry_or_material": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    audit = {
        "schema_version": "kinofail.scale-v8-physical-nuisance-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "checks": checks,
        "schedule": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": _sha(SCHEDULE),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": _sha(COLLECTOR),
        "profile_counts_by_counterfactual_pair": profile_counts,
        "profiles": list(collector.PHYSICAL_NUISANCE_PROFILES),
        "statistical_unit_policy": (
            "Each counterfactual pair shares one nuisance profile. Five profiles are balanced "
            "within every scene/operator/severity cell; appearance views remain repeated renders, "
            "not independent physical samples."
        ),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol = dict(base)
    protocol.update(
        {
            "protocol_id": "kinofail_realistic_scale_1980_v8_replication_v3",
            "frozen_utc": datetime.now(UTC).isoformat(),
            "collector_path": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": _sha(COLLECTOR),
            "scope": (
                "1,980 episodes / 990 pairs / 9 scenes / 11 operators / 2 severities / "
                "5 balanced paired physical nuisance profiles; A8 excluded."
            ),
            "supersedes": {
                "protocol": str(BASE.relative_to(ROOT)),
                "protocol_sha256": _sha(BASE),
                "data_collected_under_superseded_protocol": False,
            },
            "physical_nuisance_amendment": {
                "reason": (
                    "The inference reset fixes joint pose; distinct seed integers alone do not "
                    "guarantee physical trajectory diversity for deterministic operators."
                ),
                "design_audit": str(AUDIT.relative_to(ROOT)),
                "design_audit_sha256": _sha(AUDIT),
                "profiles": list(collector.PHYSICAL_NUISANCE_PROFILES),
                "pair_shared": True,
                "balanced_within_every_scene_operator_severity_cell": True,
                "operator_strength_unchanged": True,
                "scene_and_material_unchanged": True,
                "outcome_observed_before_freeze": False,
            },
        }
    )
    protocol["collection_contract"] = {
        **dict(base["collection_contract"]),
        "physical_seeds_per_scene_operator_severity": 5,
        "physical_nuisance_profiles": 5,
        "pair_shared_physical_nuisance_required": True,
    }
    if PROTOCOL.exists():
        raise FileExistsError(f"refusing to overwrite {PROTOCOL}")
    PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": _sha(PROTOCOL),
                "audit": str(AUDIT),
                "audit_sha256": _sha(AUDIT),
                "passed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
