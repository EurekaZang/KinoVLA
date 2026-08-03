#!/usr/bin/env python3
"""Bound irrecoverable Isaac collector hangs without retrying a pair.

This extends the sealed v1 runtime watcher to cover two additional native
failure modes observed during model-blind acquisition:

* PhysX GPU kernel-launch failures that stop before the later CUDA-700 line;
* an ``isaaclab.sh`` parent left waiting on a defunct Python child.

No label, feature, prediction, or score is read.  A pair is failed closed
when it has no summary and either (a) its log contains a fatal PhysX/CUDA
signature and has been inactive for 120 seconds, or (b) its process age
exceeds the fixed 600-second liveness deadline.  The pair is never retried.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.watch_kinofail_reconfirmation_collectors_v1 import (
    COLLECTOR_TOKEN,
    DEFAULT_EVENTS,
    FINAL_AUDIT,
    FATAL_SIGNATURES,
    _append_event,
    _argument,
    _sha256,
)


EXTENDED_FATAL_SIGNATURES = (
    *FATAL_SIGNATURES,
    "GPU convexCoreTrimeshNphase_Kernel fail to launch!!",
    "GPU prepareLostFoundPairs_Stage1 fail to launch kernel!!",
    "GPU prepareLostFoundPairs_Stage2 fail to launch kernel!!",
    "GPU updateFrictionPatches fail to launch kernel!!",
)


def _process_age_seconds(pid: int) -> float:
    stat = (Path("/proc") / str(pid) / "stat").read_text().split()
    start_ticks = int(stat[21])
    clock_ticks = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return max(0.0, uptime - start_ticks / clock_ticks)


def _process_state(pid: int) -> str | None:
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _pair_processes() -> dict[tuple[str, Path], list[dict[str, Any]]]:
    grouped: dict[tuple[str, Path], list[dict[str, Any]]] = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not tokens or COLLECTOR_TOKEN not in " ".join(tokens):
            continue
        pair_id = _argument(tokens, "--counterfactual-group-id")
        corpus_raw = _argument(tokens, "--corpus-root")
        if pair_id is None or corpus_raw is None:
            continue
        pid = int(entry.name)
        try:
            age = _process_age_seconds(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        corpus_root = Path(corpus_raw).resolve()
        grouped[(pair_id, corpus_root)].append(
            {
                "pid": pid,
                "age_seconds": age,
                "state": _process_state(pid),
                "executable": tokens[0],
                "command": tokens,
            }
        )
    return grouped


def _runtime_evidence(
    corpus_root: Path,
    pair_id: str,
    *,
    inactivity_seconds: int,
    deadline_seconds: int,
    processes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    summary = corpus_root / "pair_summaries" / f"{pair_id}.json"
    if summary.exists():
        return None
    logs = sorted(
        (corpus_root / "launcher_logs").glob(f"**/{pair_id}.log")
    )
    if len(logs) != 1:
        return None
    log = logs[0]
    try:
        tail = log.read_text(encoding="utf-8", errors="replace")[-64_000:]
        inactive_for = max(0.0, time.time() - log.stat().st_mtime)
    except FileNotFoundError:
        return None
    signatures = [
        signature
        for signature in EXTENDED_FATAL_SIGNATURES
        if signature in tail
    ]
    maximum_age = max(
        [float(process["age_seconds"]) for process in processes] + [0.0]
    )
    fatal_stall = bool(signatures) and inactive_for >= inactivity_seconds
    deadline = maximum_age >= deadline_seconds
    if not fatal_stall and not deadline:
        return None
    return {
        "runtime_log": log,
        "matched_fatal_signatures": signatures,
        "log_inactive_seconds": inactive_for,
        "maximum_process_age_seconds": maximum_age,
        "trigger": (
            "fatal_signature_and_log_inactivity"
            if fatal_stall
            else "fixed_wall_clock_liveness_deadline"
        ),
    }


def _terminate_exact_processes(
    processes: list[dict[str, Any]],
    *,
    grace_seconds: int,
) -> dict[str, Any]:
    targets = [
        int(process["pid"])
        for process in processes
        if process.get("state") not in {None, "Z"}
    ]
    term_sent: list[int] = []
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
            term_sent.append(pid)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not any(_process_state(pid) not in {None, "Z"} for pid in targets):
            break
        time.sleep(1)
    kill_sent: list[int] = []
    for pid in targets:
        if _process_state(pid) not in {None, "Z"}:
            try:
                os.kill(pid, signal.SIGKILL)
                kill_sent.append(pid)
            except ProcessLookupError:
                pass
    return {
        "observed_processes": [
            {
                "pid": int(process["pid"]),
                "state": process.get("state"),
                "age_seconds": float(process["age_seconds"]),
                "executable": process["executable"],
            }
            for process in processes
        ],
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--term-grace-seconds", type=int, default=15)
    parser.add_argument("--fatal-log-inactivity-seconds", type=int, default=120)
    parser.add_argument("--pair-deadline-seconds", type=int, default=600)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    args = parser.parse_args()
    if (
        args.poll_seconds < 5
        or args.term_grace_seconds < 1
        or args.fatal_log_inactivity_seconds < 60
        or args.pair_deadline_seconds < 300
    ):
        raise ValueError("invalid watchdog timing")
    events = args.events.resolve()
    handled: set[tuple[str, Path]] = set()
    last_report = 0.0
    while not FINAL_AUDIT.exists():
        grouped = _pair_processes()
        for key, processes in grouped.items():
            if key in handled:
                continue
            pair_id, corpus_root = key
            evidence = _runtime_evidence(
                corpus_root,
                pair_id,
                inactivity_seconds=args.fatal_log_inactivity_seconds,
                deadline_seconds=args.pair_deadline_seconds,
                processes=processes,
            )
            if evidence is None:
                continue
            handled.add(key)
            detected = datetime.now(UTC).isoformat()
            action = _terminate_exact_processes(
                processes,
                grace_seconds=args.term_grace_seconds,
            )
            log = evidence["runtime_log"]
            event = {
                "schema_version": (
                    "kinofail.reconfirmation-runtime-watchdog-event.v2"
                ),
                "detected_utc": detected,
                "completed_utc": datetime.now(UTC).isoformat(),
                "counterfactual_group_id": pair_id,
                "corpus_root": str(corpus_root),
                "runtime_log": str(log),
                "runtime_log_sha256_after_termination": _sha256(log),
                **{
                    key: value
                    for key, value in evidence.items()
                    if key != "runtime_log"
                },
                **action,
                "pair_retried": False,
                "labels_features_predictions_or_scores_read": False,
                "scientific_content_changed": False,
            }
            _append_event(events, event)
            print(json.dumps(event, sort_keys=True), flush=True)
        now = time.monotonic()
        if now - last_report >= 600:
            print(
                json.dumps(
                    {
                        "stage": "reconfirmation_runtime_watchdog_v2",
                        "active_pair_process_groups": len(grouped),
                        "handled_hung_pairs": len(handled),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_report = now
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
