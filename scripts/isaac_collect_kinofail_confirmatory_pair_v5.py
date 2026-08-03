#!/usr/bin/env python3
"""Run the audited v4 collector with a deterministic process exit.

Isaac Sim 5.1 can leave native worker threads alive after ``SimulationApp.close``
times out.  The v4 collector deliberately returns normally so acquisition
exceptions remain visible during pilot qualification.  This production
supervisor preserves that behavior, prints any exception, flushes both output
streams, and then terminates with the collector's explicit status code.

No scene, operator, sensor, policy, feature, or validation logic is changed.
"""

from __future__ import annotations

import os
import sys
import traceback

from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


def main() -> None:
    try:
        result = _load_v4_wrapper().main()
        status = int(result) if result is not None else 0
    except BaseException:  # ensure native Kit threads cannot hide the traceback
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
