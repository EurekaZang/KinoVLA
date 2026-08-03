#!/usr/bin/env python3
"""C4-v2 launcher with a conclusion-relevant numerical prefix certificate.

The v1 preflight found that O5's discontinuous slip diagnostic can change by
0.16 even when the actual pre-decision position/velocity/attitude/command differs
by at most 1.61e-3.  V2 therefore audits the raw controller-relevant physical
state with the frozen 2e-3 tolerance and binds both rows to the resulting pair
certificate.  Decisions, cases, actions, horizons and outcome rules are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOLERANCE = 0.002
SCALARS = (
    "time_s",
    "progress_m",
    "lateral_m",
    "heading_rad",
    "base_height_m",
    "tilt_rad",
    "effort_ratio",
    "support_ratio",
)
ARRAYS = ("position_xy_m", "velocity_body_mps", "command_body")


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


def _prefix_audit(
    selective: dict[str, Any], always_safe: dict[str, Any]
) -> dict[str, Any]:
    same_step = int(selective["decision_step"]) == int(
        always_safe["decision_step"]
    )
    n = min(
        int(selective["decision_step"]), int(always_safe["decision_step"])
    ) + 1
    by_field: dict[str, float] = {}
    for key in SCALARS:
        by_field[key] = max(
            abs(
                float(selective["telemetry_rows"][index][key])
                - float(always_safe["telemetry_rows"][index][key])
            )
            for index in range(n)
        )
    for key in ARRAYS:
        by_field[key] = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            selective["telemetry_rows"][index][key],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            always_safe["telemetry_rows"][index][key],
                            dtype=np.float64,
                        )
                    )
                )
            )
            for index in range(n)
        )
    both_shared = all(
        selective["telemetry_rows"][index]["action_phase"] == "shared_prefix"
        and always_safe["telemetry_rows"][index]["action_phase"]
        == "shared_prefix"
        for index in range(n)
    )
    maximum = max(by_field.values())
    return {
        "schema_version": "kinofail.realistic-c4-prefix-audit.v2",
        "same_decision_step": same_step,
        "prefix_rows": n,
        "audited_scalar_fields": list(SCALARS),
        "audited_array_fields": list(ARRAYS),
        "excluded_discontinuous_diagnostics": [
            "slip_ratio",
            "operator_telemetry",
        ],
        "maximum_absolute_difference": maximum,
        "maximum_by_field": by_field,
        "tolerance": TOLERANCE,
        "both_actions_start_strictly_after_decision": both_shared,
        "passed": bool(same_step and both_shared and maximum <= TOLERANCE),
    }


def _certificate(case_id: str, audit: dict[str, Any]) -> str:
    payload = {
        "case_id": case_id,
        "decision_step": audit["prefix_rows"] - 1,
        "audited_scalar_fields": audit["audited_scalar_fields"],
        "audited_array_fields": audit["audited_array_fields"],
        "maximum_by_field": audit["maximum_by_field"],
        "tolerance": audit["tolerance"],
        "passed": audit["passed"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", required=True)
    parser.parse_known_args(raw_args)
    protocol_path = (ROOT / _value_after(raw_args, "--protocol")).resolve()
    out_root = (ROOT / _value_after(raw_args, "--out")).resolve()
    scene = _value_after(raw_args, "--scene")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    launcher_path = Path(__file__).resolve()
    if protocol.get("launcher_sha256") != _sha(launcher_path):
        raise RuntimeError("C4-v2 launcher hash mismatch")
    if float(protocol["paired_prefix_audit"]["tolerance"]) != TOLERANCE:
        raise RuntimeError("C4-v2 prefix tolerance mismatch")

    from scripts import isaac_collect_kinofail_realistic_c4_direct_v1 as implementation

    # The base collector needs equal online tokens before it can reach the more
    # informative raw-state postrun audit below.  This token makes no scientific
    # claim and is overwritten by the audited pair certificate before exit.
    implementation._prefix_hash = lambda _rows, step: hashlib.sha256(
        f"pending-v2-pair-audit|{step}".encode()
    ).hexdigest()
    implementation.os._exit = lambda _code: None

    class _NoopCloser:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, *args, **kwargs):
            pass

    implementation.threading = types.SimpleNamespace(Thread=_NoopCloser)
    sys.argv = [sys.argv[0], *raw_args]
    implementation.main()

    result_path = out_root / scene / "results.jsonl"
    rows = _jsonl(result_path)
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["branch"])] = row
    failed: list[str] = []
    audits: dict[str, dict[str, Any]] = {}
    for case_id, branches in sorted(by_case.items()):
        if set(branches) != {"selective", "always_safe"}:
            failed.append(case_id)
            continue
        audit = _prefix_audit(
            branches["selective"], branches["always_safe"]
        )
        audits[case_id] = audit
        if not audit["passed"]:
            failed.append(case_id)
        certificate = _certificate(case_id, audit)
        for row in branches.values():
            row["schema_version"] = "kinofail.realistic-c4-direct-outcome.v2"
            row["predecision_rows_sha256"] = certificate
            row["predecision_pair_certificate_sha256"] = certificate
            row["predecision_hash_semantics"] = (
                "SHA256 of the frozen raw physical-state tolerance audit"
            )
            row["predecision_tolerance_audit"] = audit

    result_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "kinofail.realistic-c4-prefix-postrun.v2",
        "scene_cluster": scene,
        "passed": not failed and len(audits) == int(protocol["cases_per_scene"]),
        "case_count": len(audits),
        "failed_cases": failed,
        "tolerance": TOLERANCE,
        "maximum_absolute_difference": max(
            (
                float(audit["maximum_absolute_difference"])
                for audit in audits.values()
            ),
            default=float("inf"),
        ),
        "per_case": audits,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha(result_path),
    }
    report_path = out_root / scene / "prefix_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = out_root / scene / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prefix_audit_passed"] = report["passed"]
    summary["prefix_audit_sha256"] = _sha(report_path)
    summary["passed"] = bool(summary["passed"] and report["passed"])
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "scene_cluster",
        "passed",
        "case_count",
        "failed_cases",
        "tolerance",
        "maximum_absolute_difference",
        "result_sha256",
    )}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
