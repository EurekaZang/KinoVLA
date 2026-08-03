#!/usr/bin/env python3
"""Targeted O9 repair: retain frozen physics and move the ridge 0.15 m down-route.

The original universal early region starts an O9 ridge at roughly 0.23 m, where the Go2 front
feet can stop against it before the base enters the measurement region.  This amendment changes
only the ridge center from route progress 0.35 m to 0.50 m.  It is restricted to failed O9 pairs;
all geometry, severity parameters, controller, sensing, capture, and validation remain v7.
"""

from __future__ import annotations

import hashlib
import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
V4 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v4.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
EXPECTED_V4_SHA256 = "7f8956a162ffd2cda6a5207efefc1cd87805df9d48881072f555c13dc17985b0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v4():
    if _sha256(V4) != EXPECTED_V4_SHA256:
        raise RuntimeError("O9 repair v4 dependency hash mismatch")
    spec = importlib.util.spec_from_file_location("kinofail_scale_collector_v4_o9_repair", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_10hz_implementation():
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("O9 repair v1 dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    old, new = "    capture_stride = 20\n", "    capture_stride = 5\n"
    if source.count(old) != 1:
        raise RuntimeError("frozen v1 capture-stride patch point is not unique")
    module = types.ModuleType("kinofail_scale_collector_v1_10hz_o9_repair")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source.replace(old, new), str(V1), "exec"), module.__dict__)
    return module


def _install_o9_downroute_region(implementation) -> None:
    from kino_vla.utils.geometry import Rect

    prior_install = implementation._install_operator

    def shifted_region(frame):
        half_length = 0.30
        progress = min(0.50, frame.route_length_m - half_length - 0.05)
        progress = max(progress, half_length + 0.05)
        half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
        center = frame.point(progress, 0.0)
        if abs(frame.direction[0]) >= 1.0 - 1.0e-6:
            return Rect(float(center[0]), float(center[1]), half_length, half_width)
        if abs(frame.direction[1]) >= 1.0 - 1.0e-6:
            return Rect(float(center[0]), float(center[1]), half_width, half_length)
        raise ValueError("O9 repair requires an axis-aligned audited route")

    def install_operator(backend, record, frame, seed):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("targeted O9 repair refuses non-O9 records")
        original_region = implementation._region
        implementation._region = shifted_region
        try:
            return prior_install(backend, record, frame, seed)
        finally:
            implementation._region = original_region

    implementation._install_operator = install_operator


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_10hz_implementation()
    v4._install_reachable_exposure_contract(implementation)
    _install_o9_downroute_region(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
