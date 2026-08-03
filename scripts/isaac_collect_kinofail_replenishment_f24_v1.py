#!/usr/bin/env python3
"""F24 Scale collector with an explicit O4 QA-wiring correction.

The frozen v8 scientific collector and process-local RTX backend are retained.
The frozen reachable-region adapter expects a Python operator object, whereas
O4 is implemented directly in the backend and therefore returns ``None``.
F24 admits O4 when its physical telemetry records at least one attachment
cycle.  No simulation, region, force, peel, sensor, model, feature, threshold,
or analysis logic is changed.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts import isaac_collect_kinofail_confirmatory_pair_v7 as v7
from scripts import isaac_collect_kinofail_confirmatory_pair_v8 as v8
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py"
BACKEND = ROOT / "kino_vla/sim/isaac_policy_backend.py"
EXPECTED_V8_SHA256 = (
    "ccef4cd59126976c6fb9c9d6870432a4f3972ca2abb5f6edf7bae5cfe13dfa39"
)
EXPECTED_BACKEND_SHA256 = (
    "ac1f5d3fee3c938462d529a0a08f68cf93713ec60cbe32e557793709863d42a3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_f24_o4_qa_adapter(wrapper: Any) -> None:
    prior_load_v4 = wrapper._load_v4

    def load_v4() -> Any:
        module = prior_load_v4()
        prior_install = module._install_reachable_exposure_contract

        def install_reachable_exposure_contract(implementation: Any) -> None:
            prior_install(implementation)
            prior_active = implementation._active_mechanism

            def active_mechanism(
                operator_id: str,
                telemetry: dict[str, Any],
                operator: Any,
            ) -> bool:
                if operator_id == "O4_tether":
                    return int(telemetry.get("total_attachment_cycles", 0)) > 0
                return bool(prior_active(operator_id, telemetry, operator))

            implementation._active_mechanism = active_mechanism

        module._install_reachable_exposure_contract = (
            install_reachable_exposure_contract
        )
        return module

    wrapper._load_v4 = load_v4


def main() -> None:
    try:
        for path, expected in (
            (V8, EXPECTED_V8_SHA256),
            (BACKEND, EXPECTED_BACKEND_SHA256),
            (v7.V5, v7.EXPECTED_V5_SHA256),
            (v7.V6, v7.EXPECTED_V6_SHA256),
            (v7.DESIGN, v7.EXPECTED_DESIGN_SHA256),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(f"F24 exact-hash dependency mismatch: {path}")
        wrapper = _load_v4_wrapper()
        # The formal inner provenance remains bound to the frozen v5 collector;
        # the F24 freeze manifest separately binds this outer correction.
        wrapper.__file__ = str(v7.V5)
        wrapper._install_nuisance_contract = v7._install_f0_nuisance_contract
        _install_f24_o4_qa_adapter(wrapper)
        v8._install_registry_scene_source_adapter(wrapper)
        result = wrapper.main()
        status = int(result) if result is not None else 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
