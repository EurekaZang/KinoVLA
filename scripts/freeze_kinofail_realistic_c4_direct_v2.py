#!/usr/bin/env python3
"""Supersede C4-v1 after its numerical paired-prefix implementation preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "configs/data/kinofail_realistic_c4_direct_formal_v1.json"
V1_SCHEDULE = ROOT / "outputs/kinofail_realistic/design_c4_direct_v1/schedule.jsonl"
V2_SCHEDULE = ROOT / "outputs/kinofail_realistic/design_c4_direct_v2/schedule.jsonl"
V2_PROTOCOL = ROOT / "configs/data/kinofail_realistic_c4_direct_formal_v2.json"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v1.py"
LAUNCHER = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v2.py"
EVALUATOR = ROOT / "kino_vla/eval/realistic_c4.py"
V1_PREFLIGHT = (
    ROOT
    / "outputs/kinofail_realistic/corpus_c4_direct_v1/indoor_kitchen_140/results.jsonl"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(case_id: str) -> int:
    raw = f"realistic-c4-direct-v2|{case_id}".encode()
    return 300_000_000 + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_700_000_000


def main() -> int:
    if V2_SCHEDULE.exists() or V2_PROTOCOL.exists():
        raise FileExistsError("C4-v2 freeze already exists")
    v1 = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    if _sha(V1_SCHEDULE) != v1["schedule_sha256"]:
        raise RuntimeError("C4-v1 schedule provenance mismatch")
    if not V1_PREFLIGHT.is_file():
        raise RuntimeError("the excluded v1 numerical preflight is missing")
    rows = [
        json.loads(line)
        for line in V1_SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in rows:
        # Fresh reset seeds make every v2 physical consequence untouched even
        # for cases reached during the excluded v1 engineering preflight.
        row["reset_seed"] = _stable_seed(str(row["case_id"]))
        row["schema_version"] = "kinofail.realistic-c4-direct-schedule.v2"
        row["supersedes_reset_seed_from_v1"] = True
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    V2_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    V2_SCHEDULE.write_text(payload, encoding="utf-8")

    protocol = dict(v1)
    protocol.update(
        {
            "schema_version": "kinofail.realistic-c4-direct-protocol.v2",
            "protocol_id": "kinofail-realistic-c4-direct-final-v2",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": (
                "frozen_after_excluded_numerical_prefix_preflight_before_any_v2_outcome"
            ),
            "supersedes": str(V1_PROTOCOL.relative_to(ROOT)),
            "schedule": str(V2_SCHEDULE.relative_to(ROOT)),
            "schedule_sha256": _sha(V2_SCHEDULE),
            "collector": str(BASE_COLLECTOR.relative_to(ROOT)),
            "collector_sha256": _sha(BASE_COLLECTOR),
            "launcher": str(LAUNCHER.relative_to(ROOT)),
            "launcher_sha256": _sha(LAUNCHER),
            "evaluator": str(EVALUATOR.relative_to(ROOT)),
            "evaluator_sha256": _sha(EVALUATOR),
            "excluded_preflight": {
                "root": "outputs/kinofail_realistic/corpus_c4_direct_v1",
                "result_path": str(V1_PREFLIGHT.relative_to(ROOT)),
                "result_sha256": _sha(V1_PREFLIGHT),
                "counts_as_c4_evidence": False,
                "finding": (
                    "O5 has discontinuous slip-ratio noise while the largest "
                    "controller-relevant state/command difference is 0.001610 m/s"
                ),
                "decisions_actions_cases_or_outcome_rules_changed": False,
            },
            "paired_prefix_audit": {
                "mode": "raw controller-relevant physical state",
                "tolerance": 0.002,
                "required": True,
                "scalar_fields": [
                    "time_s",
                    "progress_m",
                    "lateral_m",
                    "heading_rad",
                    "base_height_m",
                    "tilt_rad",
                    "effort_ratio",
                    "support_ratio",
                ],
                "array_fields": [
                    "position_xy_m",
                    "velocity_body_mps",
                    "command_body",
                ],
                "excluded_discontinuous_diagnostics": [
                    "slip_ratio",
                    "operator_telemetry",
                ],
            },
            "v2_changes": [
                "fresh reset seed for every case",
                "raw physical-prefix tolerance certificate",
            ],
            "unchanged_from_v1": [
                "all 75 source RGB/proprio snapshots",
                "all five model checkpoints",
                "five-seed decisions and 0.5 threshold",
                "release coverage",
                "test scenes, domains, operators and nuisance realizations",
                "shared recovery actor",
                "selective and always-safe actions",
                "horizon, terminal cost and success rules",
                "acceptance criteria and bootstrap",
            ],
        }
    )
    protocol["frozen_decision_rule"] = dict(protocol["frozen_decision_rule"])
    protocol["frozen_decision_rule"]["direct_v2_test_outcomes_available_at_freeze"] = False
    V2_PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    V2_PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "passed": True,
        "protocol": str(V2_PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha(V2_PROTOCOL),
        "schedule": str(V2_SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": _sha(V2_SCHEDULE),
        "case_count": len(rows),
        "all_v2_reset_seeds_fresh": all(
            row["reset_seed"]
            != original["reset_seed"]
            for row, original in zip(
                rows,
                [
                    json.loads(line)
                    for line in V1_SCHEDULE.read_text(encoding="utf-8").splitlines()
                    if line
                ],
            )
        ),
        "v2_outcomes_existed_at_freeze": False,
        "v1_preflight_excluded": True,
    }
    (V2_SCHEDULE.parent / "freeze_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
