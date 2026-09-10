#!/usr/bin/env python3
"""Extract T2 features when the derived corpus is stored outside the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_kinofail_confirmatory_t2_features_v1 as base


def main() -> int:
    # The source extractor uses ROOT only to serialize provenance paths.  Using
    # the filesystem root preserves every absolute input while allowing /data.
    base.source.ROOT = Path("/")
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
