#!/usr/bin/env python3
"""Run F32c with the direct-v2 speed adapter and frozen F32 supervisor."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_kinofail_confirmatory_o9_pilot_f32 as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32c"
V1 = str(ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v1.py")
V2 = str(ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v2.py")


def main() -> int:
    base.OUTPUT = OUTPUT
    original_popen = base.subprocess.Popen

    def popen(command, *args, **kwargs):
        rewritten = [V2 if str(value) == V1 else value for value in command]
        return original_popen(rewritten, *args, **kwargs)

    base.subprocess.Popen = popen
    result = base.main()
    audit_path = OUTPUT / "final_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        audit["schema_version"] = "kinofail.confirmatory-o9-pilot-f32c-final-audit.v1"
        audit["pilot_variant"] = "locomotion-stable-approach-speed-0.18-mps"
        base.atomic_json(audit_path, audit)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
