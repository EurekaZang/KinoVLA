#!/usr/bin/env python3
"""Run exactly one frozen T2 case in one fresh Isaac process.

The full frozen schedule path is still passed to and authenticated by the
frozen collector.  This wrapper only limits the in-memory scene rows to the
single case named by ``KINOVLA_F21_T2_CASE_ID`` after loading the exact frozen
collector implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import isaac_collect_kinofail_confirmatory_t2_v1 as frozen


CASE_ENV = "KINOVLA_F21_T2_CASE_ID"


def select_case_rows(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("case_id")) == case_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"F21 expected one frozen schedule row for {case_id}, "
            f"found {len(selected)}"
        )
    return selected


def main() -> int:
    case_id = os.environ.get(CASE_ENV, "")
    if not case_id:
        raise RuntimeError(f"{CASE_ENV} is required")
    module = frozen._load()
    original_jsonl = module._jsonl

    def filtered_jsonl(path: Path | str) -> list[dict[str, Any]]:
        rows = original_jsonl(path)
        if any("case_id" in row for row in rows):
            return select_case_rows(rows, case_id=case_id)
        return rows

    module._jsonl = filtered_jsonl
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
