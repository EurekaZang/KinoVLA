#!/usr/bin/env python3
"""Run the fail-closed post-scale path through core models and the frozen A6 sweep."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
BASE_CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v7"
REPAIR_CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v7_o9_repair_v2"
REPAIR_PROTOCOL = ROOT / "configs/data/kinofail_realistic_o9_repair_formal_v2.json"
SNAPSHOT_V3 = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v3.json"
SNAPSHOT_V4 = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v4.json"


def _run(arguments: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command = [PYTHON, *arguments]
    print(json.dumps({"running": command}), flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def _progress() -> dict:
    completed = _run(
        ["scripts/audit_kinofail_realistic_scale_progress.py"], capture=True
    )
    result = json.loads(completed.stdout)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    progress = _progress()
    if progress["collected_episodes"] != progress["scheduled_episodes"]:
        raise RuntimeError("base scale-v7 collection is not complete")
    blockers = list(progress["blocking_repair_pairs"])
    blocker_operators = {
        str(row["operator"]) for row in progress["blocking_repair_episodes"]
    }
    if blocker_operators - {"O9_high_centering"}:
        raise RuntimeError(f"unadmitted blocker types require inspection: {sorted(blocker_operators)}")

    snapshot_protocol = SNAPSHOT_V3
    finalizer = ["scripts/finalize_kinofail_realistic_scale_v5.py"]
    if blockers:
        if not REPAIR_PROTOCOL.exists():
            _run(["scripts/freeze_kinofail_realistic_o9_repair_v2.py"])
        repair = json.loads(REPAIR_PROTOCOL.read_text(encoding="utf-8"))
        admitted = sorted(str(value) for value in repair["allowed"]["counterfactual_group_ids"])
        if admitted != sorted(blockers):
            raise RuntimeError("frozen repair-pair list no longer matches the final base audit")
        # A non-zero launcher code is allowed here: the finalizer below independently decides
        # whether any remaining runtime issue is scientifically blocking.
        _run(
            [
                "scripts/run_kinofail_realistic_scale_v2.py",
                "--schedule", "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl",
                "--collector", "scripts/isaac_collect_kinofail_realistic_o9_repair_v2.py",
                "--protocol", str(REPAIR_PROTOCOL.relative_to(ROOT)),
                "--corpus-root", str(REPAIR_CORPUS.relative_to(ROOT)),
                "--pair-ids", *admitted,
                "--audit-name", "o9_route_axis_full_v2.json",
            ],
            check=False,
        )
        if not SNAPSHOT_V4.exists():
            _run(["scripts/freeze_kinofail_realistic_snapshot_overlay_v1.py"])
        snapshot_protocol = SNAPSHOT_V4
        finalizer.extend(
            [
                "--repair-corpus-root", str(REPAIR_CORPUS.relative_to(ROOT)),
                "--repair-protocol", str(REPAIR_PROTOCOL.relative_to(ROOT)),
            ]
        )

    _run(finalizer)
    _run(
        [
            "scripts/build_kinofail_realistic_snapshot_dev.py",
            "--protocol", str(snapshot_protocol.relative_to(ROOT)),
        ]
    )
    _run(["scripts/extract_kinofail_realistic_features.py"])
    _run(["scripts/run_kinofail_realistic_a1_diagnostic_v1.py"])
    _run(["scripts/run_kinofail_realistic_multimodal_v1.py"])
    _run(
        [
            "scripts/run_kinofail_realistic_scale_v2.py",
            "--schedule", "outputs/kinofail_realistic/design_a6_boundary_v2/schedule.jsonl",
            "--collector", "scripts/isaac_collect_kinofail_realistic_pair_v6.py",
            "--protocol", "configs/data/kinofail_realistic_a6_boundary_formal_v3.json",
            "--corpus-root", "outputs/kinofail_realistic/corpus_a6_boundary_v3",
            "--audit-name", "full_collection.json",
        ],
        check=False,
    )
    _run(
        [
            "scripts/analyze_kinofail_realistic_a6_v1.py",
            "--config", "configs/eval/kinofail_realistic_a6_analysis_v2.json",
        ]
    )
    _run(
        [
            "scripts/audit_kinofail_realistic_a0_a7.py",
            "--contract", "configs/eval/kinofail_realistic_a0_a7_v4.json",
            "--out", "outputs/eval/realistic_a0_a7_v4/readiness_audit.json",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
