#!/usr/bin/env python3
"""Formal F38 runner with accepted-case-scoped temporal accounting."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/run_kinofail_t3_runin_f38_collection.py"
EXPECTED_PREDECESSOR_SHA256 = "b6516b3a485f1dd38a56d6b9b63db2d754c31db7b514754609ba12e1810e45cf"


def main() -> int:
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F38 formal runner predecessor drift")
    source = PREDECESSOR.read_text()
    old = '"all_accepted_cases_have_two_temporal_windows": len(alignments) == 2 * accepted_count,'
    new = (
        '"all_accepted_cases_have_two_temporal_windows": '
        'sum(1 for case in accepted for pair_id in case["pair_ids"] '
        'if f"{pair_id}_anomaly" in alignments) == 2 * accepted_count,'
    )
    if source.count(old) != 1:
        raise RuntimeError("F38 formal runner patch point is not unique")
    source = source.replace(old, new)
    module = types.ModuleType("kinofail_t3_runin_f38_collection_v2_implementation")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(PREDECESSOR), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
