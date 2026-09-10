#!/usr/bin/env python3
"""Freeze the severity confirmation with the nominal-posture remediation controller."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kinofail_action_severity_v1 as base


def main() -> int:
    base.OUTPUT = base.ROOT / "outputs/kinofail_action_severity_v2"
    base.COLLECTOR = "scripts/isaac_collect_kinofail_remediation_v2.py"
    base.PROTOCOL_ID = "kinofail-action-severity-v2-independent-confirmation-20260814"
    base.HORIZON_STEPS = 1200
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
