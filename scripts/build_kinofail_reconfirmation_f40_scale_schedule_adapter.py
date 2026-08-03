#!/usr/bin/env python3
"""Add the registry-equivalent scene_cluster alias for Scale evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scale_schedule.jsonl"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
F39_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f39_failure_audit/audit.json"
OUTPUT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_f40_inputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    failure = load(F39_FAILURE)
    registry = load(REGISTRY)
    scenes = {str(row["scene_id"]) for row in registry["scenes"]}
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    if (
        failure.get("passed") is not True
        or failure.get("model_prediction_truth_key_or_score_read") is not False
        or len(rows) != 21_120
        or len({str(row["counterfactual_group_id"]) for row in rows}) != 10_560
        or any("scene_cluster" in row for row in rows)
        or any(
            str(row.get("scene_family")) not in scenes
            or str(row.get("scene_family")) != str(row.get("scene_id"))
            for row in rows
        )
    ):
        raise RuntimeError("F40 Scale alias source is not the frozen model-blind design")
    adapted = [{**row, "scene_cluster": str(row["scene_family"])} for row in rows]
    by_group = Counter(str(row["counterfactual_group_id"]) for row in adapted)
    if set(by_group.values()) != {2}:
        raise RuntimeError("F40 Scale adapter changed the paired design")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "scale_schedule.jsonl"
    with schedule.open("x", encoding="utf-8") as stream:
        for row in adapted:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "schema_version": "kinofail.reconfirmation-f40-scale-schedule-adapter.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "status": "sealed_model_blind_evaluation_alias",
        "model_prediction_truth_key_or_score_read": False,
        "scientific_record_or_value_changed": False,
        "only_added_key": "scene_cluster",
        "alias_rule": "scene_cluster := scene_family",
        "counts": {
            "rows": len(adapted),
            "pairs": len(by_group),
            "scenes": len({row["scene_cluster"] for row in adapted}),
            "operators": len({row["target_operator"] for row in adapted}),
        },
        "checks": {
            "all_scene_aliases_registry_backed": all(
                row["scene_cluster"] in scenes for row in adapted
            ),
            "all_other_fields_byte_semantically_equal": all(
                {key: value for key, value in target.items() if key != "scene_cluster"}
                == source
                for source, target in zip(rows, adapted, strict=True)
            ),
            "pair_cardinality_preserved": set(by_group.values()) == {2},
        },
        "source_sha256": {
            "source_schedule": sha256(SOURCE),
            "scene_registry": sha256(REGISTRY),
            "f39_failure_audit": sha256(F39_FAILURE),
            "builder": sha256(Path(__file__).resolve()),
        },
        "output": str(schedule.relative_to(ROOT)),
        "output_sha256": sha256(schedule),
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
