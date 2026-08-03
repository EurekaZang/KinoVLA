#!/usr/bin/env python3
"""Seal F29 over only case-complete pairs untouched by F28."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F27 = ROOT / "outputs/kinofail_t3_replenishment_f27"
F28 = ROOT / "outputs/kinofail_t3_replenishment_f28"
OUTPUT = ROOT / "outputs/kinofail_t3_replenishment_f29"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f29/corpus")


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F29 seal or corpus")
    f27_path = F27 / "freeze_manifest.json"
    f27 = read_json(f27_path)
    if (
        (F27 / "freeze_manifest.sha256").read_text(encoding="utf-8").split()[0] != sha256(f27_path)
        or f27.get("passed") is not True
        or f27.get("counterfactual_pairs") != 182
    ):
        raise RuntimeError("F27 freeze invalid")
    interruption_path = F28 / "interruption_audit.json"
    interruption = read_json(interruption_path)
    if (
        interruption.get("passed") is not True
        or interruption.get("attempted_pair_count") != 13
        or interruption.get("passed_summary_count") != 0
        or interruption.get("accepted_pair_count") != 0
    ):
        raise RuntimeError("F28 interruption audit invalid")

    attempted = set(str(value) for value in interruption["attempted_pair_ids"])
    mappings = read_jsonl(ROOT / str(f27["mapping"]))
    eligible = [
        row
        for row in mappings
        if str(row["replacement_o7_group_id"]) not in attempted
        and str(row["replacement_o8_group_id"]) not in attempted
    ]
    excluded = [row for row in mappings if row not in eligible]
    eligible_pairs = sorted(
        pair_id
        for row in eligible
        for pair_id in (
            str(row["replacement_o7_group_id"]),
            str(row["replacement_o8_group_id"]),
        )
    )
    if len(eligible) != 80 or len(excluded) != 11 or len(eligible_pairs) != 160:
        raise RuntimeError("unexpected F29 untouched-case selection")
    if attempted.intersection(eligible_pairs):
        raise RuntimeError("F29 contains an F28-attempted pair")

    eligible_path = OUTPUT / "design/eligible_cases.jsonl"
    excluded_path = OUTPUT / "design/excluded_f28_touched_cases.jsonl"
    pair_path = OUTPUT / "design/eligible_pair_ids.jsonl"
    write_jsonl(eligible_path, eligible)
    write_jsonl(excluded_path, excluded)
    write_jsonl(pair_path, [{"counterfactual_group_id": value} for value in eligible_pairs])

    dependencies = [
        ROOT / "scripts/run_kinofail_t3_replenishment_f29.py",
        ROOT / "scripts/run_kinofail_t3_replenishment_f28.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py",
        ROOT / "kino_vla/sim/isaac_policy_backend.py",
        Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh"),
    ]
    seal = {
        "schema_version": "kinofail.t3-replenishment-f29-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_collection",
        "passed": True,
        "scientific_design_changed_from_f27": False,
        "selection_basis": "exclude whole case when either frozen pair was touched by F28",
        "selection_depends_on_pass_fail_outcome": False,
        "f28_artifact_reuse": False,
        "result_dependent_retry": False,
        "model_prediction_feature_or_score_read": False,
        "maximum_concurrent_isaac_processes": 3,
        "pilot_health_gate_pairs": 3,
        "pilot_abort_condition": "all three return nonzero without pair summaries",
        "eligible_cases": 80,
        "excluded_f28_touched_cases": 11,
        "counterfactual_pairs": 160,
        "physical_episodes": 320,
        "minimum_complete_cases_for_strictly_below_five_percent": 17,
        "corpus_root": str(CORPUS),
        "eligible_case_mapping": str(eligible_path.relative_to(ROOT)),
        "eligible_case_mapping_sha256": sha256(eligible_path),
        "excluded_case_mapping": str(excluded_path.relative_to(ROOT)),
        "excluded_case_mapping_sha256": sha256(excluded_path),
        "eligible_pair_ids": str(pair_path.relative_to(ROOT)),
        "eligible_pair_ids_sha256": sha256(pair_path),
        "f27_freeze": str(f27_path.relative_to(ROOT)),
        "f27_freeze_sha256": sha256(f27_path),
        "f28_interruption_audit": str(interruption_path.relative_to(ROOT)),
        "f28_interruption_audit_sha256": sha256(interruption_path),
        "inherited_schedules": f27["schedules"],
        "inherited_protocols": f27["protocols"],
        "scene_registry": f27["scene_registry"],
        "scene_registry_sha256": f27["scene_registry_sha256"],
        "material_lock": f27["material_lock"],
        "material_lock_sha256": f27["material_lock_sha256"],
        "dependencies": [
            {"path": str(path), "sha256": sha256(path)} for path in dependencies
        ],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(
        f"{sha256(seal_path)}  {seal_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
