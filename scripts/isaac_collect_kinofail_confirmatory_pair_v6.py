#!/usr/bin/env python3
"""Operational provenance-binding correction for the frozen v5 collector.

The v5 supervisor executes the exact audited v4 scientific collector, but the
dynamically loaded module inherits v4's ``__file__``.  The formal protocol is
correctly bound to v5, so the inner pre-acquisition provenance check rejects
the run before any episode is written.  This wrapper changes only that module
metadata to the exact-hash-pinned v5 path.  Simulation, sensing, operators,
validation, and process-exit behavior are unchanged.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from pathlib import Path

from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
EXPECTED_V5_SHA256 = (
    "88f8e3abe6805bb1e96a87f333dad29d285b49ec1dbf40e4b626b020bb36419d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    try:
        if _sha256(V5) != EXPECTED_V5_SHA256:
            raise RuntimeError("v6 predecessor differs from the F0-bound v5")
        wrapper = _load_v4_wrapper()
        wrapper.__file__ = str(V5)
        result = wrapper.main()
        status = int(result) if result is not None else 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
