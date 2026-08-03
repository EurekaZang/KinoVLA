#!/usr/bin/env python3
"""Targeted O9 repair: rotate the rounded ridge with the audited route axis.

Scale-v7 originally authored every rounded O9 cylinder with a world-Y axis.  That is correct
for world-X routes, but turns the same object into a longitudinal runner with a vertical end cap
on world-Y routes.  All six O9 QA failures are world-Y cases: the robot stops at that end cap
before the scheduled region and never exposes the high-centering mechanism.

This amendment changes only the cylinder axis for world-Y routes.  It retains the frozen early
region, ridge height/width, physics material, controller, sensing, appearance views, and 10 Hz
capture path.  The correction is selected from geometry/QA telemetry, never model predictions.
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
        raise RuntimeError("O9 repair v2 dependency hash mismatch: scale collector v4")
    spec = importlib.util.spec_from_file_location("kinofail_scale_collector_v4_o9_repair_v2", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_10hz_implementation():
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("O9 repair v2 dependency hash mismatch: scale collector v1")
    source = V1.read_text(encoding="utf-8")
    old, new = "    capture_stride = 20\n", "    capture_stride = 5\n"
    if source.count(old) != 1:
        raise RuntimeError("frozen v1 capture-stride patch point is not unique")
    module = types.ModuleType("kinofail_scale_collector_v1_10hz_o9_repair_v2")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source.replace(old, new), str(V1), "exec"), module.__dict__)
    return module


def _route_aligned_geometry_kind(frame, requested: str) -> str:
    """Return the backend cylinder orientation that is transverse to the route."""
    if requested != "rounded_ridge":
        return requested
    if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
        return "rounded_ridge"  # Cylinder axis Y, transverse to a world-X route.
    if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
        return "longitudinal_rounded_ridge"  # Cylinder axis X, transverse to world-Y.
    raise ValueError("O9 repair v2 requires an axis-aligned audited route")


def _install_route_aligned_o9(implementation) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend, record, frame, seed):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("targeted O9 repair v2 refuses non-O9 records")
        if record["condition"] != "anomaly":
            return prior_install(backend, record, frame, seed)

        import kino_vla.sim.operators as operators

        original = operators.HighCentering

        def route_aligned_high_centering(
            region,
            residual_support=0.15,
            *,
            ridge_height_m=None,
            ridge_width_m=None,
            geometry_kind="box_ridge",
        ):
            return original(
                region,
                residual_support,
                ridge_height_m=ridge_height_m,
                ridge_width_m=ridge_width_m,
                geometry_kind=_route_aligned_geometry_kind(frame, geometry_kind),
            )

        operators.HighCentering = route_aligned_high_centering
        try:
            return prior_install(backend, record, frame, seed)
        finally:
            operators.HighCentering = original

    implementation._install_operator = install_operator


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_10hz_implementation()
    v4._install_reachable_exposure_contract(implementation)
    _install_route_aligned_o9(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
