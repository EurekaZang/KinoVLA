#!/usr/bin/env python3
"""Development-only A4 replay with an explicit locomotion-policy override.

This wrapper exists only to isolate the suspected low-level-policy mismatch
before a new C4 protocol is frozen.  It deliberately reuses the historical A4
scenes, seeds, operators, action definitions, and outcome code, but it does not
claim confirmatory status and it never writes into the historical A4 root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--development-policy", required=True)
    parser.add_argument("--development-audit", required=True)
    dev_args, forwarded = parser.parse_known_args()
    out_root = (ROOT / _value_after(forwarded, "--out")).resolve()
    scene = _value_after(forwarded, "--scene")
    protocol_path = (ROOT / _value_after(forwarded, "--protocol")).resolve()
    policy_path = (ROOT / dev_args.development_policy).resolve()
    audit_path = (ROOT / dev_args.development_audit).resolve()
    historical_root = (
        ROOT / "outputs/kinofail_realistic/corpus_a4_actual_action_v5"
    ).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    if out_root == historical_root or historical_root in out_root.parents:
        raise RuntimeError("development wrapper refuses to write into historical A4 root")

    from kino_vla.utils import config as config_module

    original_load_config = config_module.load_config
    relative_policy = str(policy_path.relative_to(ROOT))
    load_calls: list[dict[str, Any]] = []

    def load_config_with_policy(
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ):
        combined = dict(overrides or {})
        if str(path) == "sim/go2_skeleton.yaml":
            combined["policy_path"] = relative_policy
            load_calls.append(
                {
                    "config": str(path),
                    "policy_path": relative_policy,
                    "policy_sha256": _sha(policy_path),
                }
            )
        return original_load_config(path, combined)

    config_module.load_config = load_config_with_policy
    sys.argv = [sys.argv[0], *forwarded]

    from scripts import isaac_collect_kinofail_realistic_a4_action_v5 as launcher

    # The historical one-shot collector uses os._exit to guarantee Kit teardown,
    # which intentionally suppresses Python tracebacks and prevents this wrapper
    # from writing its development audit. Disable only inside this disposable
    # process; the original source file and formal collector hash stay untouched.
    launcher.implementation.os._exit = lambda _code: None
    class _NoopCloser:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, *args, **kwargs):
            pass

    launcher.implementation.threading = types.SimpleNamespace(Thread=_NoopCloser)
    launcher.implementation.main()

    result_path = out_root / scene / "results.jsonl"
    rows = _jsonl(result_path)
    actions = sorted({str(row["action"]) for row in rows})
    case_ids = sorted({str(row["case_id"]) for row in rows})
    report = {
        "schema_version": "kinofail.realistic-policy-swap-development-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_complete",
        "passed": (
            len(rows) == 2 * len(case_ids)
            and all(
                len([row for row in rows if row["case_id"] == case_id]) == 2
                and len(
                    {
                        "continue" if row["action"] == "continue" else "recovery"
                        for row in rows
                        if row["case_id"] == case_id
                    }
                )
                == 2
                for case_id in case_ids
            )
            and len(load_calls) == 1
        ),
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "scene_cluster": scene,
        "case_count": len(case_ids),
        "episode_count": len(rows),
        "actions": actions,
        "policy_override": {
            "path": relative_policy,
            "sha256": _sha(policy_path),
            "load_calls": load_calls,
        },
        "frozen_historical_inputs_reused_for_diagnosis": {
            "protocol_path": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha(protocol_path),
            "schedule_path": json.loads(protocol_path.read_text())["schedule"],
            "scene_cluster": scene,
        },
        "implementation": {
            "wrapper_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "wrapper_sha256": _sha(Path(__file__).resolve()),
            "historical_collector_path": "scripts/isaac_collect_kinofail_realistic_a4_action_v1.py",
            "historical_collector_sha256": _sha(
                ROOT / "scripts/isaac_collect_kinofail_realistic_a4_action_v1.py"
            ),
        },
        "result": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": _sha(result_path),
        },
        "interpretation_boundary": (
            "This replay isolates the locomotion policy on a historical development "
            "design. It can diagnose the policy mismatch, but it cannot validate C4: "
            "there is no direct selective-versus-always-safe branch and the scenes "
            "are not an untouched confirmation split."
        ),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
