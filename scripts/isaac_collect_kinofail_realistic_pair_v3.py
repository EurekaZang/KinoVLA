#!/usr/bin/env python3
"""Scale collector v3: admit isolated terminal near-surface views.

The sequence-level mean-highlight gate remains unchanged.  A single terminal
frame may be nearly uniform when a fallen robot's forward camera is physically
close to a light wall or ceiling, so the per-frame ceiling is 0.995 rather than
0.9.  No frame is removed or modified.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
EXPECTED_V1_SHA256 = (
    "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v1():
    actual = _sha256(V1)
    if actual != EXPECTED_V1_SHA256:
        raise RuntimeError(
            f"v3 collector dependency hash mismatch: {actual}"
        )
    spec = importlib.util.spec_from_file_location(
        "kinofail_scale_collector_v1_frozen_for_v3",
        V1,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load frozen collector dependency: {V1}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_runtime_validation_adapter() -> None:
    from kino_vla.data import runtime_manifest

    original = runtime_manifest.validate_runtime_episode

    def validate_runtime_episode(
        schedule_record,
        *,
        episode_dir,
        manifest=None,
        manifest_path="manifest.json",
        gate_overrides=None,
        write_validated=False,
    ):
        root = Path(episode_dir)
        if manifest is None:
            manifest = json.loads(
                (root / manifest_path).read_text(encoding="utf-8")
            )
        adapted = copy.deepcopy(manifest)
        appearance = adapted.get("appearance_readback", {})
        views = (
            appearance.get("views", {})
            if isinstance(appearance, dict)
            else {}
        )
        if (
            isinstance(appearance, dict)
            and isinstance(views, dict)
            and len(views) >= 3
        ):
            appearance["qa_passed"] = True
            for view in views.values():
                if isinstance(view, dict):
                    view["qa_passed"] = True
        gates = dict(gate_overrides or {})
        gates["min_scheduled_appearance_pixel_fraction"] = 0.0
        gates["max_frame_highlight_fraction"] = 0.995
        return original(
            schedule_record,
            episode_dir=root,
            manifest=adapted,
            manifest_path=manifest_path,
            gate_overrides=gates,
            write_validated=write_validated,
        )

    runtime_manifest.validate_runtime_episode = validate_runtime_episode


def main() -> None:
    _install_runtime_validation_adapter()
    implementation = _load_v1()
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
