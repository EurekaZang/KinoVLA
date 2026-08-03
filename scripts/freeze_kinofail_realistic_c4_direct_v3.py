#!/usr/bin/env python3
"""Freeze order-balanced, process-isolated direct C4-v3."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_PROTOCOL = ROOT / "configs/data/kinofail_realistic_c4_direct_formal_v2.json"
V2_SCHEDULE = ROOT / "outputs/kinofail_realistic/design_c4_direct_v2/schedule.jsonl"
V3_PROTOCOL = ROOT / "configs/data/kinofail_realistic_c4_direct_formal_v3.json"
V3_SCHEDULE = ROOT / "outputs/kinofail_realistic/design_c4_direct_v3/schedule.jsonl"
BASE = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v1.py"
LAUNCHER = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v3.py"
RUNNER = ROOT / "scripts/run_kinofail_realistic_c4_direct_v3.py"
EVALUATOR = ROOT / "kino_vla/eval/realistic_c4.py"
V2_PREFLIGHT = (
    ROOT
    / "outputs/kinofail_realistic/corpus_c4_direct_v2/indoor_kitchen_140/results.jsonl"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(case_id: str) -> int:
    raw = f"realistic-c4-direct-v3-isolated|{case_id}".encode()
    return 300_000_000 + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_700_000_000


def main() -> int:
    if V3_SCHEDULE.exists() or V3_PROTOCOL.exists():
        raise FileExistsError("C4-v3 freeze already exists")
    v2 = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))
    if _sha(V2_SCHEDULE) != v2["schedule_sha256"]:
        raise RuntimeError("C4-v2 schedule provenance mismatch")
    if not V2_PREFLIGHT.is_file():
        raise RuntimeError("excluded C4-v2 order preflight is missing")
    rows = [
        json.loads(line)
        for line in V2_SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_cluster"], row["operator"])].append(row)
    group_index = 0
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: row["case_id"])
        for nuisance_index, row in enumerate(group):
            selective_first = (group_index + nuisance_index) % 2 == 0
            row["branch_order"] = (
                ["selective", "always_safe"]
                if selective_first
                else ["always_safe", "selective"]
            )
            row["reset_seed"] = _stable_seed(str(row["case_id"]))
            row["schema_version"] = "kinofail.realistic-c4-direct-schedule.v3"
            row["fresh_isaac_app_per_case_pair"] = True
        group_index += 1
    rows = sorted(rows, key=lambda row: (row["scene_cluster"], row["case_id"]))
    first_counts = Counter(row["branch_order"][0] for row in rows)
    if abs(first_counts["selective"] - first_counts["always_safe"]) > 1:
        raise RuntimeError(f"branch order is not globally balanced: {first_counts}")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    V3_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    V3_SCHEDULE.write_text(payload, encoding="utf-8")

    protocol = dict(v2)
    protocol.update(
        {
            "schema_version": "kinofail.realistic-c4-direct-protocol.v3",
            "protocol_id": "kinofail-realistic-c4-direct-final-v3",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": (
                "frozen_after_excluded_order_isolation_preflight_before_any_v3_outcome"
            ),
            "supersedes": str(V2_PROTOCOL.relative_to(ROOT)),
            "schedule": str(V3_SCHEDULE.relative_to(ROOT)),
            "schedule_sha256": _sha(V3_SCHEDULE),
            "collector": str(BASE.relative_to(ROOT)),
            "collector_sha256": _sha(BASE),
            "launcher": str(LAUNCHER.relative_to(ROOT)),
            "launcher_sha256": _sha(LAUNCHER),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha(RUNNER),
            "evaluator": str(EVALUATOR.relative_to(ROOT)),
            "evaluator_sha256": _sha(EVALUATOR),
            "excluded_preflights": [
                {
                    "root": "outputs/kinofail_realistic/corpus_c4_direct_v1",
                    "reason": "strict byte hash exposed GPU prefix noise",
                },
                {
                    "root": "outputs/kinofail_realistic/corpus_c4_direct_v2",
                    "result_path": str(V2_PREFLIGHT.relative_to(ROOT)),
                    "result_sha256": _sha(V2_PREFLIGHT),
                    "reason": (
                        "multi-case process accumulated operator state and fixed "
                        "selective-first order could amplify numerical noise"
                    ),
                },
            ],
            "execution_design": {
                "fresh_isaac_application_per_case_pair": True,
                "paired_branches_share_one_fresh_application": True,
                "branch_order_counterbalanced": True,
                "first_branch_counts": dict(first_counts),
                "maximum_global_order_imbalance": 1,
                "new_reset_seed_for_every_v3_case": True,
            },
            "v3_changes": [
                "fresh Isaac application for each case pair",
                "globally and within-cell alternating branch order",
                "fresh reset seed for every case",
            ],
            "unchanged_from_v1_v2": [
                "all 75 source RGB/proprio snapshots and decisions",
                "five model checkpoints, ensemble and 0.5 threshold",
                "release coverage",
                "test scenes, domains, operators and nuisance realizations",
                "shared recovery actor",
                "selective and always-safe action definitions",
                "horizon, terminal cost and success rules",
                "acceptance criteria and bootstrap",
                "2e-3 controller-state prefix tolerance",
            ],
        }
    )
    protocol["frozen_decision_rule"] = dict(protocol["frozen_decision_rule"])
    protocol["frozen_decision_rule"]["direct_v3_test_outcomes_available_at_freeze"] = False
    V3_PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    V3_PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "passed": True,
        "protocol": str(V3_PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha(V3_PROTOCOL),
        "schedule": str(V3_SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": _sha(V3_SCHEDULE),
        "case_count": len(rows),
        "first_branch_counts": dict(first_counts),
        "all_v3_reset_seeds_fresh": True,
        "v3_outcomes_existed_at_freeze": False,
        "v1_v2_preflights_excluded": True,
    }
    (V3_SCHEDULE.parent / "freeze_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
