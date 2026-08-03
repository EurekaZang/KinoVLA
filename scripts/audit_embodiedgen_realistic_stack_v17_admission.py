#!/usr/bin/env python3
"""Bind the v16 stack admission to the operator-blind route-motion visual proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-v16-audit", type=Path, required=True)
    parser.add_argument("--motion-proxy-audit", type=Path, required=True)
    parser.add_argument("--route-compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stack_path = args.stack_v16_audit.resolve()
    motion_path = args.motion_proxy_audit.resolve()
    route_path = args.route_compiled_audit.resolve()
    stack = _json(stack_path)
    motion = _json(motion_path)
    route = _json(route_path)
    scene_ids = {stack.get("scene_id"), motion.get("scene_id"), route.get("scene_id")}
    checks = {
        "v16_stack_passed": stack.get("passed") is True,
        "motion_proxy_passed": motion.get("passed") is True,
        "motion_proxy_operator_blind": motion.get("operator_blind") is True,
        "motion_proxy_binds_route": motion.get("compiled_audit_sha256") == _sha256(route_path),
        "scene_id_consistent": len(scene_ids) == 1 and None not in scene_ids,
        "development_boundary_retained": motion.get("counts_as_a0_a7_evidence") is False
        and stack.get("counts_as_a0_a7_evidence") is False,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-stack-v17-admission.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": route.get("scene_id"),
        "passed": passed,
        "checks": checks,
        "evidence": {
            "stack_v16": {"path": str(stack_path), "sha256": _sha256(stack_path)},
            "motion_proxy": {"path": str(motion_path), "sha256": _sha256(motion_path)},
            "route_compiled": {"path": str(route_path), "sha256": _sha256(route_path)},
        },
        "admission_state": (
            "realistic_stack_v17_admitted_for_operator_confirmation"
            if passed
            else "realistic_stack_v17_rejected"
        ),
        "counts_as_realistic_corpus_evidence": False,
        "counts_as_a0_a7_evidence": False,
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
