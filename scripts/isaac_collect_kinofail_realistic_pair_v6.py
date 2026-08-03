#!/usr/bin/env python3
"""Scale-v6 collector: reachable exposure contract with 10 Hz synchronized RTX capture."""

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
        raise RuntimeError("scale-v6 v4 dependency hash mismatch")
    spec = importlib.util.spec_from_file_location("kinofail_scale_collector_v4_frozen", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_10hz_implementation():
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("scale-v6 v1 dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    old, new = "    capture_stride = 20\n", "    capture_stride = 5\n"
    if source.count(old) != 1:
        raise RuntimeError("frozen v1 capture-stride patch point is not unique")
    module = types.ModuleType("kinofail_scale_collector_v1_10hz")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source.replace(old, new), str(V1), "exec"), module.__dict__)
    return module


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_10hz_implementation()
    v4._install_reachable_exposure_contract(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
