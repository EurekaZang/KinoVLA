#!/usr/bin/env python3
"""Run F32b through the sealed F32 model-blind pilot supervisor."""

from __future__ import annotations

from pathlib import Path

from scripts import run_kinofail_confirmatory_o9_pilot_f32 as base


def main() -> int:
    base.OUTPUT = Path(__file__).resolve().parents[1] / "outputs/kinofail_confirmatory_o9_pilot_f32b"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
