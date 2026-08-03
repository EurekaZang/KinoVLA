#!/usr/bin/env python3
"""Run the unchanged realistic-stack admission logic under the v16 schema label."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.audit_embodiedgen_realistic_stack_v15_admission as implementation


if __name__ == "__main__":
    implementation.SCHEMA_VERSION = "kinofail.embodiedgen-realistic-stack-v16-admission.v1"
    raise SystemExit(implementation.main())
