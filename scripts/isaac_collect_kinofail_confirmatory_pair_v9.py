#!/usr/bin/env python3
"""Bind the v8 collector to the process-local RTX texture implementation.

The v8 scientific collector is unchanged.  This supervisor exact-hash pins
the only operational backend correction used by F12: generated fallback
textures live under a process-specific directory so concurrent Isaac
processes cannot overwrite a PNG while RTX is reading it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from pathlib import Path

from scripts import isaac_collect_kinofail_confirmatory_pair_v8 as v8


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py"
BACKEND = ROOT / "kino_vla/sim/isaac_policy_backend.py"
EXPECTED_V8_SHA256 = (
    "ccef4cd59126976c6fb9c9d6870432a4f3972ca2abb5f6edf7bae5cfe13dfa39"
)
EXPECTED_BACKEND_SHA256 = (
    "ac1f5d3fee3c938462d529a0a08f68cf93713ec60cbe32e557793709863d42a3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    try:
        for path, expected in (
            (V8, EXPECTED_V8_SHA256),
            (BACKEND, EXPECTED_BACKEND_SHA256),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(
                    f"v9 exact-hash dependency mismatch: {path}"
                )
        v8.main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
