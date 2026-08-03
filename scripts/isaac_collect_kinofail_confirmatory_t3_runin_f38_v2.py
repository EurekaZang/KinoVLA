#!/usr/bin/env python3
"""F38 run-in v2: freeze the model-blind feasible contact interval."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import isaac_collect_kinofail_confirmatory_t3_runin_f38 as base


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38.py"
EXPECTED_PREDECESSOR_SHA256 = "2236ca71c30ca8196872150bbe1c81a5394d36c6aba72648bf19e42b9f857ef4"


def main() -> None:
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F38-v2 predecessor collector drift")
    # Near edge = 0.82 - 0.30 = 0.52 m.  With the frozen 0.35 m
    # footprint margin, the earliest encounter boundary is 0.17 m: beyond
    # every frozen nuisance start (<=0.135 m), but within the minimum
    # development-pilot nominal reach (0.5296 m).
    base.RUNIN_CENTER_PROGRESS_M = 0.82
    base.RUNIN_HALF_LENGTH_M = 0.30
    base.main()


if __name__ == "__main__":
    main()
