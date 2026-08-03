#!/usr/bin/env python3
"""Restore omitted T3 scene-source readback without changing frozen records.

The T3 schedule was frozen without the redundant ``scene_source`` key even
though the same value is already hash-locked in the scene registry.  The v7
scientific collector reads that key only while constructing
``scene_readback``.  This supervisor supplies the registry value through a
dict view whose canonical JSON content remains byte-for-byte equivalent to
the original schedule record.  Consequently:

* the frozen schedule and its per-record hashes remain unchanged;
* simulation, sensing, operators, validation, features, and thresholds remain
  unchanged;
* schedules that already contain ``scene_source`` follow the exact v7 path.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from kino_vla.data.runtime_manifest import schedule_record_sha256
from scripts import isaac_collect_kinofail_confirmatory_pair_v7 as v7
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v7.py"
EXPECTED_V7_SHA256 = (
    "de790876ff90331ab40fdbc85f90ee691594cda07dad4ec9222af285999aa726"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RegistryBackedSceneSourceRecord(dict[str, Any]):
    """Expose one registry-backed readback key without changing JSON content."""

    def __init__(self, record: dict[str, Any], scene_source: str) -> None:
        super().__init__(record)
        self._scene_source = scene_source

    def __getitem__(self, key: str) -> Any:
        if key == "scene_source" and key not in self:
            return self._scene_source
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "scene_source" and key not in self:
            return self._scene_source
        return super().get(key, default)


def _install_registry_scene_source_adapter(wrapper: Any) -> None:
    prior_loader = wrapper._load_confirmatory_implementation

    def load_implementation() -> Any:
        implementation = prior_loader()
        prior_collect_one = implementation._collect_one

        def collect_one(
            backend: Any,
            record: dict[str, Any],
            registry_row: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if "scene_source" in record:
                return prior_collect_one(
                    backend, record, registry_row, *args, **kwargs
                )
            if (
                record.get("battery") != "c2_t3"
                or record.get("scene_family") != registry_row.get("scene_id")
                or record.get("source_scene_id")
                != registry_row.get("source_scene_id")
            ):
                raise RuntimeError(
                    "scene-source fallback is not authorized for this record"
                )
            scene_source = registry_row.get("source")
            if not isinstance(scene_source, str) or not scene_source:
                raise RuntimeError(
                    "scene registry has no non-empty source provenance"
                )
            adapted = _RegistryBackedSceneSourceRecord(
                record, scene_source
            )
            if schedule_record_sha256(adapted) != schedule_record_sha256(
                record
            ):
                raise RuntimeError(
                    "scene-source adapter changed the frozen record hash"
                )
            return prior_collect_one(
                backend, adapted, registry_row, *args, **kwargs
            )

        implementation._collect_one = collect_one
        return implementation

    wrapper._load_confirmatory_implementation = load_implementation


def main() -> None:
    try:
        for path, expected in (
            (V7, EXPECTED_V7_SHA256),
            (v7.V5, v7.EXPECTED_V5_SHA256),
            (v7.V6, v7.EXPECTED_V6_SHA256),
            (v7.DESIGN, v7.EXPECTED_DESIGN_SHA256),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(
                    f"v8 exact-hash dependency mismatch: {path}"
                )
        wrapper = _load_v4_wrapper()
        wrapper.__file__ = str(v7.V5)
        wrapper._install_nuisance_contract = (
            v7._install_f0_nuisance_contract
        )
        _install_registry_scene_source_adapter(wrapper)
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
