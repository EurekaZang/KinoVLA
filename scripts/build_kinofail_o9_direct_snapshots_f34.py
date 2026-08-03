#!/usr/bin/env python3
"""Extract F33 snapshots at the first strict direct high-centering event."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import kino_vla.data.realistic_snapshots as snapshots
from kino_vla.data.o9_semantics import evaluate_o9_high_centering
from scripts import build_kinofail_realistic_snapshot_dev as base


def first_direct_o9_event(rows: Sequence[Mapping[str, Any]]) -> float:
    for row in rows:
        value = row.get("operator")
        telemetry = value if isinstance(value, Mapping) else row
        if evaluate_o9_high_centering(telemetry)["passed"]:
            timestamp = float(row["timestamp_s"])
            if math.isfinite(timestamp):
                return timestamp
    raise ValueError("telemetry has no strict direct O9 semantic event")


def main() -> int:
    snapshots._EVENT_ADAPTERS["O9_high_centering"] = first_direct_o9_event
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
