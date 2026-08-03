#!/usr/bin/env python3
"""Fail closed on irrecoverably stopped reconfirmation collectors.

The monitor reads only Isaac/PhysX runtime logs.  It never reads labels,
features, predictions, scores, or endpoint outcomes, and it never launches or
retries a counterfactual pair.  A collector is terminated only when its own
log contains an explicit fatal CUDA/PhysX stop signature and the pair has not
produced a summary.  The ordinary launcher then records the attempt as a
terminal infrastructure attrition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVENT_ROOT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/operational_watchdog"
)
DEFAULT_EVENTS = EVENT_ROOT / "events.jsonl"
FINAL_AUDIT = (
    ROOT
    / "outputs/eval/unified_moe_v3_reconfirmation_v2/"
    "finalization_audit.json"
)
COLLECTOR_TOKEN = "isaac_collect_kinofail_confirmatory_pair_v7.py"
FATAL_SIGNATURES = (
    "simulation will be stopped and new cuda context manager will be created",
    "Synchronizing GPU Narrowphase failed! 700",
    "Fetching GPU Narrowphase failed! 700",
    "cuEventRecord failed with error 700",
    "cuStreamWaitEvent failed with error 700",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argument(tokens: list[str], name: str) -> str | None:
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _collector_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
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
        if (
            not tokens
            or COLLECTOR_TOKEN not in " ".join(tokens)
            or Path(tokens[0]).name != "python"
        ):
            continue
        pair_id = _argument(tokens, "--counterfactual-group-id")
        corpus_raw = _argument(tokens, "--corpus-root")
        if pair_id is None or corpus_raw is None:
            continue
        processes.append(
            {
                "pid": int(entry.name),
                "pair_id": pair_id,
                "corpus_root": Path(corpus_raw).resolve(),
                "command": tokens,
            }
        )
    return processes


def _fatal_log(corpus_root: Path, pair_id: str) -> tuple[Path, list[str]] | None:
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
    except FileNotFoundError:
        return None
    matched = [signature for signature in FATAL_SIGNATURES if signature in tail]
    return (log, matched) if matched else None


def _alive(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().split()[2]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return state != "Z"


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--term-grace-seconds", type=int, default=15)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.term_grace_seconds < 1:
        raise ValueError("invalid watchdog timing")
    events = args.events.resolve()
    last_report = 0.0
    handled: set[tuple[int, str]] = set()
    while not FINAL_AUDIT.exists():
        for process in _collector_processes():
            key = (int(process["pid"]), str(process["pair_id"]))
            if key in handled:
                continue
            fatal = _fatal_log(
                process["corpus_root"],
                str(process["pair_id"]),
            )
            if fatal is None:
                continue
            log, signatures = fatal
            handled.add(key)
            pid = int(process["pid"])
            detected = datetime.now(UTC).isoformat()
            try:
                os.kill(pid, signal.SIGTERM)
                action = "sigterm"
            except ProcessLookupError:
                action = "already_exited"
            deadline = time.monotonic() + args.term_grace_seconds
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(1)
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
                action = "sigterm_then_sigkill"
            event = {
                "schema_version": (
                    "kinofail.reconfirmation-runtime-watchdog-event.v1"
                ),
                "detected_utc": detected,
                "completed_utc": datetime.now(UTC).isoformat(),
                "pid": pid,
                "counterfactual_group_id": process["pair_id"],
                "corpus_root": str(process["corpus_root"]),
                "runtime_log": str(log),
                "runtime_log_sha256_after_termination": _sha256(log),
                "matched_fatal_signatures": signatures,
                "action": action,
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
                        "stage": "reconfirmation_runtime_watchdog",
                        "active_collectors": len(_collector_processes()),
                        "handled_fatal_collectors": len(handled),
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
