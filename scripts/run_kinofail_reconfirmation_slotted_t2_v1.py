#!/usr/bin/env python3
"""Run the original T2 scene launcher through the shared three-slot broker."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_t2_scene_v2 as base
from scripts.kinofail_reconfirmation_slot_pool_v1 import (
    connect_slot_pool,
)


def main() -> int:
    manager, slot_pool = connect_slot_pool()
    original_run = base.subprocess.run

    def slotted_run(*args: Any, **kwargs: Any) -> Any:
        ticket = int(slot_pool.acquire())
        print(
            json.dumps(
                {
                    "event": "isaac_slot_acquired",
                    "acquired_utc": datetime.now(UTC).isoformat(),
                    "capacity": 3,
                    "battery": "c2_t2",
                    "ticket": ticket,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            return original_run(*args, **kwargs)
        finally:
            slot_pool.release(ticket)
            print(
                json.dumps(
                    {
                        "event": "isaac_slot_released",
                        "released_utc": datetime.now(UTC).isoformat(),
                        "capacity": 3,
                        "battery": "c2_t2",
                        "ticket": ticket,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    base.subprocess.run = slotted_run
    # Keep the client manager referenced until the T2 Isaac process returns.
    _ = manager
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
