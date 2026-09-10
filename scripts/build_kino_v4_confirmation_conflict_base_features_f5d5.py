#!/usr/bin/env python3
"""Build F0b conflict features from the certified F5b2 task audit."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kino_v4_confirmation_conflict_base_features_f0b as base


def main() -> int:
    base.TASK_AUDIT = (
        ROOT
        / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"
        / "f5_task_aligned_audit_f5b2"
    )
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
