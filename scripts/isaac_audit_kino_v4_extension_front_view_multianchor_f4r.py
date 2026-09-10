#!/usr/bin/env python3
"""Run the frozen F4i front-view audit at a registered route-progress anchor."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/isaac_audit_kino_v4_extension_front_view_v1.py"
EXPECTED_PREDECESSOR_SHA256 = (
    "4cfde9787ac02430be28842c1a9ad7cdf0822af30822e889aa2ae45dc94a54a9"
)


def main() -> int:
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F4r front-view predecessor drift")
    source = PREDECESSOR.read_text(encoding="utf-8")
    replacements = {
        '    preliminary.add_argument("--minimum-swap-l1", type=float, default=0.015)\n': (
            '    preliminary.add_argument("--minimum-swap-l1", type=float, default=0.015)\n'
            '    preliminary.add_argument("--route-progress-m", type=float, required=True)\n'
        ),
        '            route.frame.point(0.0, 0.0),\n': (
            '            route.frame.point(args.route_progress_m, 0.0),\n'
        ),
        '            "minimum_swap_l1": args.minimum_swap_l1,\n': (
            '            "minimum_swap_l1": args.minimum_swap_l1,\n'
            '            "route_progress_m": args.route_progress_m,\n'
        ),
        '            "schema_version": "kinofail.kino-v4-extension-front-view-audit.v1",\n': (
            '            "schema_version": "kinofail.kino-v4-extension-front-view-multianchor-f4r-audit.v1",\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"F4r patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kino_v4_extension_front_view_multianchor_f4r")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    exec(compile(source, str(PREDECESSOR), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
