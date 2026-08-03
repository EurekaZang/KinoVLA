#!/usr/bin/env python3
"""Scale-v7 collector with the admitted route-aligned O9 rounded ridge."""

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v4():
    if _sha(V4) != EXPECTED_V4_SHA256:
        raise RuntimeError("scale-v7 dependency mismatch: v4")
    spec = importlib.util.spec_from_file_location("kinofail_scale_v7_v4", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_10hz_implementation():
    if _sha(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("scale-v7 dependency mismatch: v1")
    source = V1.read_text(encoding="utf-8")
    old, new = "    capture_stride = 20\n", "    capture_stride = 5\n"
    if source.count(old) != 1:
        raise RuntimeError("scale-v7 capture patch point is not unique")
    module = types.ModuleType("kinofail_scale_v7_10hz")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source.replace(old, new), str(V1), "exec"), module.__dict__)
    return module


def _install_route_aligned_o9(implementation) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend, record, frame, seed):
        if record["target_operator"] != "O9_high_centering" or record["condition"] != "anomaly":
            return prior_install(backend, record, frame, seed)
        if record["physical_realization"] != "rounded_ridge":
            raise RuntimeError("scale-v7 O9 alignment expects rounded_ridge")

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
            if geometry_kind != "rounded_ridge":
                raise RuntimeError(f"unexpected O9 geometry {geometry_kind}")
            if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                actual_kind = "rounded_ridge"
            elif abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                actual_kind = "longitudinal_rounded_ridge"
            else:
                raise ValueError("scale-v7 requires an axis-aligned audited route")
            return original(
                region,
                residual_support,
                ridge_height_m=ridge_height_m,
                ridge_width_m=ridge_width_m,
                geometry_kind=actual_kind,
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
