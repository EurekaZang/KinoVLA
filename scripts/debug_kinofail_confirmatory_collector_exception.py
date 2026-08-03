#!/usr/bin/env python3
"""Expose exceptions hidden by the legacy collector's unconditional os._exit."""

from __future__ import annotations

import os
import sys
import traceback
import atexit
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import main


if __name__ == "__main__":
    trace_rows: list[dict[str, object]] = []
    live_trace_path = ROOT / "outputs/debug_confirmatory_collector_live_trace.jsonl"
    live_trace_path.parent.mkdir(parents=True, exist_ok=True)
    live_trace = live_trace_path.open("w", encoding="utf-8")

    def _trace(frame, event, arg):
        filename = str(frame.f_code.co_filename)
        in_scope = (
            "/home/eureka/KinoVLA/" in filename
            or "/home/eureka/IsaacLab/source/" in filename
        )
        if in_scope and event in {"call", "return", "exception"}:
            row = {
                "event": event,
                "file": filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
            if event == "exception":
                row["exception_type"] = getattr(arg[0], "__name__", str(arg[0]))
                row["exception"] = repr(arg[1])
            trace_rows.append(row)
            del trace_rows[:-200]
            live_trace.write(json.dumps(row, sort_keys=True) + "\n")
            live_trace.flush()
        return _trace

    def _write_trace() -> None:
        live_trace.close()
        output = ROOT / "outputs/debug_confirmatory_collector_trace.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(trace_rows, indent=2) + "\n", encoding="utf-8")

    atexit.register(_write_trace)
    sys.settrace(_trace)

    def _debug_exit(status: int) -> None:
        print(f"[debug] intercepted os._exit({status})", file=sys.stderr, flush=True)

    os._exit = _debug_exit  # type: ignore[assignment]
    try:
        result = main()
        print(f"[debug] collector returned {result!r}", file=sys.stderr, flush=True)
    except BaseException:
        traceback.print_exc()
        raise
