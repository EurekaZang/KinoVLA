#!/usr/bin/env python3
"""Unattended process-only watchdog for the three-process F38 successor."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/watch_kinofail_t3_runin_f38_formal.py"
EXPECTED_PREDECESSOR_SHA256 = "60ee5efc6aa1540993606dcb542f72612bb0ffd7d475f39e3860781c590577a0"


def main() -> int:
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F38-S1 watchdog predecessor drift")
    source = PREDECESSOR.read_text()
    replacements = {
        'SEAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/seal_manifest.json"': 'SEAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal_s1/seal_manifest.json"',
        'AMENDMENT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_watchdog_amendment1/amendment_manifest.json"': 'AMENDMENT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_s1_watchdog_amendment1/amendment_manifest.json"',
        'OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_watchdog"': 'OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_s1_watchdog"',
        'FINAL = ROOT / "outputs/kinofail_t3_runin_f38_formal/final_audit.json"': 'FINAL = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"F38-S1 watchdog patch point not unique: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_t3_runin_f38_s1_watchdog_implementation")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(PREDECESSOR), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
