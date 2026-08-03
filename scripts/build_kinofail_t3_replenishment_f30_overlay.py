#!/usr/bin/env python3
"""Build a no-copy T3 union from original valid cases and accepted F29 cases.

Every accepted F29 case replaces its original invalid case as one complete
four-episode unit.  All other original cases remain in the 1,500-case design.
Schedule records are never edited, so each runtime manifest continues to match
the exact record against which it was collected.  The corpus is a symlink farm
and therefore duplicates no RGB or proprioception payloads.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_kinofail_confirmatory_valid_conflict_design_v1 import (  # noqa: E402
    _validate_t3,
)


ORIGINAL = ROOT / "outputs/kinofail_reconfirmation_v2"
F27 = ROOT / "outputs/kinofail_t3_replenishment_f27"
F29 = ROOT / "outputs/kinofail_t3_replenishment_f29"
F29_CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f29/corpus")
OUTPUT = ROOT / "outputs/eval/unified_moe_v3_t3_replenishment_f30"
DESIGN = OUTPUT / "combined_design"
UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f30/union")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    return result


def main() -> int:
    if OUTPUT.exists() or UNION.exists():
        raise FileExistsError("refusing to overwrite F30 design or union")
    audit_path = F29 / "final_audit.json"
    f29 = read_json(audit_path)
    if (
        f29.get("passed") is not True
        or f29.get("model_prediction_feature_or_score_read") is not False
        or f29.get("result_dependent_selection_or_retry") is not False
        or int(f29.get("counts", {}).get("accepted_complete_cases", 0)) < 17
    ):
        raise RuntimeError("F29 did not pass its frozen T3 case gate")
    accepted = list(f29["accepted_cases"])
    accepted_by_original = {str(row["original_case_id"]): row for row in accepted}
    if len(accepted_by_original) != len(accepted):
        raise RuntimeError("duplicate F29 replacement mapping")

    original_design = ORIGINAL / "schedules/c2_t3"
    original_cases = read_jsonl(original_design / "case_schedule.jsonl")
    original_schedule = read_jsonl(original_design / "schedule.jsonl")
    original_by_group = group_rows(original_schedule)
    original_case_ids = {str(row["case_id"]) for row in original_cases}
    if len(original_cases) != 1_500 or not set(accepted_by_original).issubset(original_case_ids):
        raise RuntimeError("F29 mapping is outside original 1,500-case design")

    replacement_cases = {
        str(row["case_id"]): row
        for row in read_jsonl(F27 / "design/case_schedule.jsonl")
    }
    replacement_schedule: list[dict[str, Any]] = []
    f27_freeze = read_json(F27 / "freeze_manifest.json")
    for artifact in f27_freeze["schedules"]:
        path = ROOT / str(artifact["path"])
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"F27 schedule drift: {path}")
        replacement_schedule.extend(read_jsonl(path))
    replacement_by_group = group_rows(replacement_schedule)

    combined_cases: list[dict[str, Any]] = []
    combined_schedule: list[dict[str, Any]] = []
    replacement_group_ids: set[str] = set()
    provenance_rows: list[dict[str, Any]] = []
    for original_case in original_cases:
        original_case_id = str(original_case["case_id"])
        mapping = accepted_by_original.get(original_case_id)
        if mapping is None:
            combined_cases.append(original_case)
            groups = [
                str(original_case["o7_source_physics_group_id"]),
                str(original_case["o8_source_physics_group_id"]),
            ]
            source = "original"
            selected_case_id = original_case_id
            lookup = original_by_group
        else:
            selected_case_id = str(mapping["replacement_case_id"])
            case = replacement_cases[selected_case_id]
            combined_cases.append(case)
            groups = [
                str(mapping["replacement_o7_group_id"]),
                str(mapping["replacement_o8_group_id"]),
            ]
            replacement_group_ids.update(groups)
            source = "f29_accepted_replacement"
            lookup = replacement_by_group
        for group_id in groups:
            rows = lookup.get(group_id, [])
            if len(rows) != 2:
                raise RuntimeError(f"F30 source pair is incomplete: {group_id}")
            combined_schedule.extend(rows)
        provenance_rows.append(
            {
                "original_case_id": original_case_id,
                "selected_case_id": selected_case_id,
                "source": source,
                "selected_group_ids": groups,
            }
        )
    combined_cases.sort(key=lambda row: str(row["case_id"]))
    combined_schedule.sort(
        key=lambda row: (str(row["counterfactual_group_id"]), str(row["condition"]))
    )
    if (
        len(combined_cases) != 1_500
        or len({str(row["case_id"]) for row in combined_cases}) != 1_500
        or len(combined_schedule) != 6_000
        or len({str(row["counterfactual_group_id"]) for row in combined_schedule}) != 3_000
    ):
        raise RuntimeError("F30 combined design cardinality mismatch")

    schedule_path = DESIGN / "schedule.jsonl"
    case_path = DESIGN / "case_schedule.jsonl"
    provenance_path = DESIGN / "case_provenance.jsonl"
    write_jsonl(schedule_path, combined_schedule)
    write_jsonl(case_path, combined_cases)
    write_jsonl(provenance_path, provenance_rows)
    combined_case_by_id = {
        str(row["case_id"]): row for row in combined_cases
    }
    replacement_id_by_original = {
        original: str(mapping["replacement_case_id"])
        for original, mapping in accepted_by_original.items()
    }
    conflict_schedule = read_jsonl(ORIGINAL / "schedules/conflict_schedule.jsonl")
    combined_conflict: list[dict[str, Any]] = []
    for source_row in conflict_schedule:
        row = json.loads(json.dumps(source_row))
        original_case_id = str(row["case_id"])
        replacement_case_id = replacement_id_by_original.get(original_case_id)
        if replacement_case_id is not None:
            replacement_case = combined_case_by_id[replacement_case_id]
            row.update(
                {
                    "benchmark_id": "kinofail_t3_replenishment_f30",
                    "case_id": replacement_case_id,
                    "case_seed": replacement_case["case_seed"],
                    "physical_nuisance": replacement_case["physical_nuisance"],
                }
            )
        combined_conflict.append(row)
    if (
        len(combined_conflict) != 18_000
        or len({str(row["case_id"]) for row in combined_conflict}) != 3_000
        or set(Counter(str(row["case_id"]) for row in combined_conflict).values()) != {6}
    ):
        raise RuntimeError("F30 combined conflict schedule cardinality mismatch")
    conflict_schedule_path = OUTPUT / "combined_conflict_schedule.jsonl"
    write_jsonl(conflict_schedule_path, combined_conflict)
    design_audit_path = DESIGN / "audit.json"
    write_json(
        design_audit_path,
        {
            "schema_version": "kinofail.t3-replenishment-f30-combined-design.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": True,
            "model_prediction_feature_or_score_read": False,
            "complete_case_replacement": True,
            "schedule_records_edited": False,
            "planned_cases": 1_500,
            "planned_physics_pairs": 3_000,
            "accepted_replacement_cases": len(accepted),
            "schedule_sha256": sha256(schedule_path),
            "case_schedule_sha256": sha256(case_path),
            "case_provenance_sha256": sha256(provenance_path),
            "combined_conflict_schedule": str(conflict_schedule_path.relative_to(ROOT)),
            "combined_conflict_schedule_sha256": sha256(conflict_schedule_path),
            "source_sha256": {
                "original_schedule": sha256(original_design / "schedule.jsonl"),
                "original_case_schedule": sha256(original_design / "case_schedule.jsonl"),
                "f29_final_audit": sha256(audit_path),
                "f27_freeze": sha256(F27 / "freeze_manifest.json"),
            },
        },
    )

    inventory: list[dict[str, Any]] = []
    UNION.mkdir(parents=True, exist_ok=False)
    for row in combined_schedule:
        relative = Path(str(row["required_outputs"]["episode_manifest"])).parent
        scene_id = str(row["scene_cluster"])
        group_id = str(row["counterfactual_group_id"])
        source_root = F29_CORPUS if group_id in replacement_group_ids else ORIGINAL / "conflict_capsules_ext4"
        source = source_root / scene_id / relative
        destination = UNION / scene_id / relative
        manifest = source / "manifest.json"
        record = {
            "episode_id": str(row["episode_id"]),
            "counterfactual_group_id": group_id,
            "source": str(source),
            "destination": str(destination),
            "source_kind": "f29" if group_id in replacement_group_ids else "original",
            "manifest_exists": manifest.is_file(),
            "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        }
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=True)
        inventory.append(record)
    inventory_path = OUTPUT / "artifact_inventory.jsonl"
    write_jsonl(inventory_path, inventory)

    _, _, valid_ids, invalid = _validate_t3(design_dir=DESIGN, corpus=UNION)
    expected_valid = 1_409 + len(accepted)
    if len(valid_ids) != expected_valid or len(invalid) != 1_500 - expected_valid:
        raise RuntimeError(
            f"F30 realized validity mismatch: valid={len(valid_ids)} invalid={len(invalid)} expected={expected_valid}"
        )
    final = read_json(design_audit_path)
    final.update(
        {
            "status": "validated_model_blind_union",
            "valid_cases": len(valid_ids),
            "invalid_case_count": len(invalid),
            "case_attrition_rate": len(invalid) / 1_500,
            "strictly_below_five_percent": len(invalid) / 1_500 < 0.05,
            "union_root": str(UNION),
            "symlink_payload_copy": False,
            "artifact_inventory": str(inventory_path.relative_to(ROOT)),
            "artifact_inventory_sha256": sha256(inventory_path),
            "valid_case_ids_sha256": hashlib.sha256(
                "\n".join(valid_ids).encode("utf-8")
            ).hexdigest(),
            "invalid_case_ledger": invalid,
        }
    )
    if final["strictly_below_five_percent"] is not True:
        raise RuntimeError("F30 T3 case attrition gate failed")
    write_json(design_audit_path, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
