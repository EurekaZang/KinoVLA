#!/usr/bin/env python3
"""Build fresh O7/O8 physical pairs for C2 v5 confirmation."""

from __future__ import annotations

import build_kinofail_realistic_c2_v3_t3_schedule as implementation


implementation.DESIGN_TAG = (
    "c2_bidirectional_confirmation_v5_t3_a1_strong_swaps"
)
_base_views = implementation._views


def _strong_views(*args, **kwargs):
    """Keep both nuisance views materially distinct from the primary."""

    views = _base_views(*args, **kwargs)
    source = views[1]
    target = views[2]
    for key in (
        "material_family",
        "material_asset_id",
        "material_semantics",
        "material_source",
        "material_license",
        "physical_size_m",
    ):
        target[key] = source[key]
    return views


implementation._views = _strong_views


if __name__ == "__main__":
    raise SystemExit(implementation.main())
