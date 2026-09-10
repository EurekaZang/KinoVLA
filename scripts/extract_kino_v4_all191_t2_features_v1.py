#!/usr/bin/env python3
"""Run the frozen T2 extractor on an absolute-path all191 corpus union."""

from __future__ import annotations

from pathlib import Path

from scripts import extract_kinofail_realistic_c1_causal_features_v1 as base


def main() -> int:
    # The frozen extractor uses ROOT only when serializing provenance paths.
    # Filesystem root keeps absolute /data inputs representable without
    # changing any image, proprioception, or feature computation.
    base.ROOT = Path("/")
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
