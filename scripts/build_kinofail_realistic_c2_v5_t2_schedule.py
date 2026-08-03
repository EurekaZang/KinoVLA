#!/usr/bin/env python3
"""Build the fresh three-scene T2 arm for C2 v5 confirmation."""

from __future__ import annotations

import build_kinofail_realistic_c2_v3_t2_schedule as implementation


implementation.DESIGN_TAG = "c2_bidirectional_confirmation_v5_t2"


if __name__ == "__main__":
    raise SystemExit(implementation.main())
