#!/usr/bin/env python3
"""Run/resume and directly audit the frozen 90-pair O9 semantic repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import evaluate_o9_high_centering


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
LAUNCHER = ROOT / "scripts/run_kinofail_realistic_scale_v2.py"
SCHEDULE = (
    ROOT / "outputs/kinofail_realistic/design_o9_semantic_repair_v4/schedule.jsonl"
)
REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
PROTOCOL = (
    ROOT / "configs/data/kinofail_realistic_o9_semantic_repair_formal_v4.json"
)
COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v4.py"
)
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_o9_semantic_repair_v4"
AUDIT = CORPUS / "semantic_repair_audit.json"
PROTOCOL_ID = "kinofail_realistic_o9_semantic_repair_v4"
ALLOWED_RUNTIME_ISSUE_SUFFIXES = (
    "appearance_effect_too_small",
    "rgb_spatial_contrast_too_low",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def command(pair_ids: list[str], audit_name: str) -> list[str]:
    return [
        str(PYTHON),
        str(LAUNCHER),
        "--schedule",
        str(SCHEDULE),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(PROTOCOL),
        "--corpus-root",
        str(CORPUS),
        "--collector",
        str(COLLECTOR),
        "--pair-ids",
        *pair_ids,
        "--audit-name",
        audit_name,
    ]


def allowed_runtime_issues(issues: Any) -> bool:
    return isinstance(issues, list) and all(
        any(str(issue).endswith(suffix) for suffix in ALLOWED_RUNTIME_ISSUE_SUFFIXES)
        for issue in issues
    )


def audit_pair(pair_id: str) -> tuple[bool, list[str], dict[str, Any]]:
    summary_path = CORPUS / "pair_summaries" / f"{pair_id}.json"
    summary = load_json(summary_path)
    results = summary.get("results")
    issues: list[str] = []
    evidence: dict[str, Any] = {
        "summary": str(summary_path),
        "summary_present": bool(summary),
    }
    if not isinstance(results, list) or len(results) != 2:
        return False, ["missing_or_incomplete_pair_summary"], evidence
    by_condition = {str(row.get("condition")): row for row in results}
    if set(by_condition) != {"anomaly", "nominal_counterfactual"}:
        issues.append("condition_pair_incomplete")
        return False, issues, evidence

    manifests = {}
    nuisances = []
    for condition, result in by_condition.items():
        manifest = load_json(Path(str(result.get("manifest", ""))))
        manifests[condition] = manifest
        if not manifest:
            issues.append(f"manifest_missing:{condition}")
            continue
        formal = manifest.get("collection", {}).get("formal_protocol", {})
        if formal.get("protocol_id") != PROTOCOL_ID:
            issues.append(f"formal_protocol_mismatch:{condition}")
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if not isinstance(nuisance, dict):
            issues.append(f"physical_nuisance_missing:{condition}")
        else:
            nuisances.append(nuisance)
        runtime_issues = manifest.get("runtime_validation", {}).get("issues", [])
        if not allowed_runtime_issues(runtime_issues):
            issues.extend(
                f"runtime:{condition}:{value}" for value in runtime_issues
            )

    nominal_result = by_condition["nominal_counterfactual"]
    if nominal_result.get("fallen") is not False:
        issues.append("nominal_fell")
    if float(nominal_result.get("max_route_deviation_m", 1.0e9)) > 0.55:
        issues.append("nominal_left_audited_route")

    semantic: dict[str, Any] = {}
    anomaly = manifests.get("anomaly", {})
    if anomaly:
        semantic = evaluate_o9_high_centering(
            anomaly.get("operator_readback", {}).get("telemetry", {})
        )
        if not semantic["passed"]:
            issues.append("direct_o9_semantic_gate_failed")
    if len(nuisances) != 2 or nuisances[0] != nuisances[1]:
        issues.append("counterfactual_physical_nuisance_mismatch")

    evidence.update(
        {
            "nominal_fallen": nominal_result.get("fallen"),
            "nominal_max_route_deviation_m": nominal_result.get(
                "max_route_deviation_m"
            ),
            "anomaly_semantic": semantic,
            "pair_shared_nuisance": nuisances[0] if len(nuisances) == 2 else None,
        }
    )
    return not issues, sorted(set(issues)), evidence


def write_audit(
    *,
    pair_ids: list[str],
    returncodes: list[int],
    running: bool,
) -> dict[str, Any]:
    checks = {pair_id: audit_pair(pair_id) for pair_id in pair_ids}
    failures = {
        pair_id: issues
        for pair_id, (passed, issues, _) in checks.items()
        if not passed
    }
    report = {
        "schema_version": "kinofail.o9-semantic-repair-runtime-audit.v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "running"
            if running
            else ("complete" if not failures else "complete_with_exclusions")
        ),
        "protocol_id": PROTOCOL_ID,
        "pairs_scheduled": len(pair_ids),
        "pairs_audited_pass": len(pair_ids) - len(failures),
        "returncodes": returncodes,
        "failures": failures,
        "per_pair": {
            pair_id: evidence for pair_id, (_, _, evidence) in checks.items()
        },
        "acceptance": {
            "nominal_must_not_fall": True,
            "maximum_nominal_route_deviation_m": 0.55,
            "anomaly_must_pass_direct_o9_semantics": True,
            "counterfactual_nuisance_must_match": True,
            "allowed_nonphysical_runtime_issue_suffixes": list(
                ALLOWED_RUNTIME_ISSUE_SUFFIXES
            ),
        },
        "a8_in_scope": False,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    protocol = load_json(PROTOCOL)
    pair_ids = [
        str(value) for value in protocol["allowed"]["counterfactual_group_ids"]
    ]
    if len(pair_ids) != 90:
        raise RuntimeError(f"expected 90 frozen O9 pairs, got {len(pair_ids)}")

    completed = {
        pair_id for pair_id in pair_ids if audit_pair(pair_id)[0]
    }
    remaining = [pair_id for pair_id in pair_ids if pair_id not in completed]
    shards = [remaining[index :: args.shards] for index in range(args.shards)]
    shards = [shard for shard in shards if shard]
    print(
        json.dumps(
            {
                "status": "starting",
                "scheduled": len(pair_ids),
                "already_accepted": len(completed),
                "remaining": len(remaining),
                "shards": [len(shard) for shard in shards],
            }
        ),
        flush=True,
    )
    if not remaining:
        report = write_audit(pair_ids=pair_ids, returncodes=[], running=False)
        print(json.dumps(report, indent=2))
        return 0

    processes = []
    for index, shard in enumerate(shards):
        cmd = command(shard, f"full_collection_shard{index}.json")
        processes.append(subprocess.Popen(cmd, cwd=ROOT))
    write_audit(pair_ids=pair_ids, returncodes=[], running=True)
    returncodes = [process.wait() for process in processes]
    report = write_audit(
        pair_ids=pair_ids,
        returncodes=returncodes,
        running=False,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pairs_audited_pass": report["pairs_audited_pass"],
                "returncodes": returncodes,
                "failures": report["failures"],
                "audit": str(AUDIT),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
