#!/usr/bin/env python3
"""Collect v16 O1/O2/O3 lanes with continuous O2 support and failure-safe exit."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isaac_collect_realistic_route_operator_lane_v6 import main


if __name__ == "__main__":
    try:
        code = main(
            controller_mode="damped_v3",
            collector_lifecycle="native_v16_continuous_o2_dependency_closed",
            collector_script=Path(__file__).resolve(),
        )
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
