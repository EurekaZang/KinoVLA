#!/usr/bin/env python3
"""Final C4 launcher: one isolated Isaac app per counterbalanced case pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v1.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_after(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"required forwarded argument missing: {flag}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_isolated_implementation() -> types.ModuleType:
    source = BASE.read_text(encoding="utf-8")
    replacements = [
        (
            '    preliminary.add_argument("--scene", required=True)\n',
            '    preliminary.add_argument("--scene", required=True)\n'
            '    preliminary.add_argument("--case-id", required=True)\n',
        ),
        (
            '    parser.add_argument("--scene", default=pre.scene)\n',
            '    parser.add_argument("--scene", default=pre.scene)\n'
            '    parser.add_argument("--case-id", default=pre.case_id)\n',
        ),
        (
            '        cases = [\n'
            '            row for row in _jsonl(schedule_path) if row["scene_cluster"] == args.scene\n'
            '        ]\n'
            '        expected_per_scene = int(protocol["cases_per_scene"])\n',
            '        cases = [\n'
            '            row for row in _jsonl(schedule_path)\n'
            '            if row["scene_cluster"] == args.scene and row["case_id"] == args.case_id\n'
            '        ]\n'
            '        expected_per_scene = 1\n',
        ),
        (
            '            for branch in ("selective", "always_safe"):\n',
            '            for branch in case["branch_order"]:\n',
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"C4-v3 source patch point is not unique: {old!r}")
        source = source.replace(old, new, 1)
    module = types.ModuleType("kinofail_c4_direct_isolated_v3")
    module.__file__ = str(BASE)
    module.__package__ = "scripts"
    exec(compile(source, str(BASE), "exec"), module.__dict__)
    return module


def main() -> int:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--case-id", required=True)
    parser.parse_known_args(raw_args)
    protocol_path = (ROOT / _value_after(raw_args, "--protocol")).resolve()
    out_root = (ROOT / _value_after(raw_args, "--out")).resolve()
    scene = _value_after(raw_args, "--scene")
    case_id = _value_after(raw_args, "--case-id")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("launcher_sha256") != _sha(Path(__file__).resolve()):
        raise RuntimeError("C4-v3 launcher hash mismatch")
    if protocol.get("collector_sha256") != _sha(BASE):
        raise RuntimeError("C4-v3 base collector hash mismatch")

    from scripts.isaac_collect_kinofail_realistic_c4_direct_v2 import (
        _certificate,
        _prefix_audit,
    )

    implementation = _load_isolated_implementation()
    implementation._prefix_hash = lambda _rows, step: hashlib.sha256(
        f"pending-v3-isolated-pair-audit|{step}".encode()
    ).hexdigest()
    implementation.os._exit = lambda _code: None
    sys.argv = [sys.argv[0], *raw_args]
    implementation.main()

    result_path = out_root / scene / "results.jsonl"
    rows = _jsonl(result_path)
    current = [row for row in rows if str(row["case_id"]) == case_id]
    by_branch = {str(row["branch"]): row for row in current}
    if set(by_branch) != {"selective", "always_safe"} or len(current) != 2:
        raise RuntimeError(f"incomplete isolated case: {case_id}")
    audit = _prefix_audit(
        by_branch["selective"], by_branch["always_safe"]
    )
    certificate = _certificate(case_id, audit)
    for row in current:
        row["schema_version"] = "kinofail.realistic-c4-direct-outcome.v3"
        row["predecision_rows_sha256"] = certificate
        row["predecision_pair_certificate_sha256"] = certificate
        row["predecision_hash_semantics"] = (
            "SHA256 of the frozen isolated raw physical-state tolerance audit"
        )
        row["predecision_tolerance_audit"] = audit
        row["branch_order"] = list(
            next(
                item["branch_order"]
                for item in _jsonl(
                    (ROOT / protocol["schedule"]).resolve()
                )
                if item["case_id"] == case_id
            )
        )
        row["branch_order_index"] = row["branch_order"].index(row["branch"])
        row["fresh_isaac_app_per_case_pair"] = True

    current_by_branch = {row["branch"]: row for row in current}
    updated = [
        (
            current_by_branch[str(row["branch"])]
            if str(row["case_id"]) == case_id
            else row
        )
        for row in rows
    ]
    result_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in updated),
        encoding="utf-8",
    )

    audit_dir = out_root / scene / "prefix_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{case_id}.json"
    audit_payload = {
        "case_id": case_id,
        "scene_cluster": scene,
        "passed": audit["passed"],
        "audit": audit,
        "certificate_sha256": certificate,
        "result_path": str(result_path.relative_to(ROOT)),
    }
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    schedule_rows = [
        row
        for row in _jsonl((ROOT / protocol["schedule"]).resolve())
        if row["scene_cluster"] == scene
    ]
    all_rows = _jsonl(result_path)
    complete_case_ids = {
        value
        for value in {str(row["case_id"]) for row in all_rows}
        if {
            str(row["branch"])
            for row in all_rows
            if str(row["case_id"]) == value
        }
        == {"selective", "always_safe"}
    }
    audit_files = list(audit_dir.glob("*.json"))
    audit_values = [
        json.loads(path.read_text(encoding="utf-8")) for path in audit_files
    ]
    summary = {
        "schema_version": "kinofail.realistic-c4-direct-scene-summary.v3",
        "scene_cluster": scene,
        "protocol_id": protocol["protocol_id"],
        "passed": (
            len(complete_case_ids) == len(schedule_rows)
            and len(audit_values) == len(schedule_rows)
            and all(bool(value["passed"]) for value in audit_values)
        ),
        "expected_case_pairs": len(schedule_rows),
        "completed_case_pairs": len(complete_case_ids),
        "passed_prefix_audits": sum(
            bool(value["passed"]) for value in audit_values
        ),
        "fresh_isaac_app_per_case_pair": True,
        "counterbalanced_branch_order": True,
        "weak_null_harmful_and_fallen_outcomes_retained": True,
    }
    (out_root / scene / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_id": case_id,
                "scene_cluster": scene,
                "prefix_audit_passed": audit["passed"],
                "prefix_maximum_absolute_difference": audit[
                    "maximum_absolute_difference"
                ],
                "completed_case_pairs": summary["completed_case_pairs"],
                "expected_case_pairs": summary["expected_case_pairs"],
                "scene_complete": summary["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
