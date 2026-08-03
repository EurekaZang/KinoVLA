#!/usr/bin/env python3
"""Confirmatory collector with a physically advanced post-failure sensor window.

This wrapper exact-hash loads the excluded-pilot v1 collector.  Before the
first detected fall it is identical to v1.  After a fall, it advances Isaac
Sim for 25 additional control steps through ``step_terminal_sensing`` with a
zero velocity command.  The terminal label remains latched, while PhysX, RTX,
IMU, articulation state, and operator forces continue to update.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V1_WRAPPER = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py"
EXPECTED_V1_WRAPPER_SHA256 = (
    "dd3aba9fc758b2de0cf6589b86aecb5e1396c3bf51c817d1bd5261293828d47d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v3_wrapper() -> Any:
    if _sha256(V1_WRAPPER) != EXPECTED_V1_WRAPPER_SHA256:
        raise RuntimeError("confirmatory v3 dependency differs from excluded pilot")
    source = V1_WRAPPER.read_text(encoding="utf-8")
    marker = (
        '        "    capture_stride = 20\\n": '
        '"    capture_stride = 5\\n",\n'
    )
    insertion = marker + r'''        "    while step < 360 and not obs.fallen:\n": (
            "    first_fall_step = None\n"
            "    while step < 360:\n"
        ),
        "        obs = backend.step(command)\n": (
            "        if first_fall_step is None:\n"
            "            obs = backend.step(command)\n"
            "        else:\n"
            "            obs = backend.step_terminal_sensing()\n"
        ),
        "        step += 1\n\n    np.savez_compressed(\n": (
            "        step += 1\n"
            "        if obs.fallen and first_fall_step is None:\n"
            "            first_fall_step = step\n"
            "        if (\n"
            "            first_fall_step is not None\n"
            "            and step - first_fall_step >= 25\n"
            "            and len(features) >= 25\n"
            "            and len(rgb[primary_id]) >= 5\n"
            "        ):\n"
            "            break\n\n"
            "    np.savez_compressed(\n"
        ),
'''
    if source.count(marker) != 1:
        raise RuntimeError("v3 fall-horizon insertion point is not unique")
    source = source.replace(marker, insertion)

    metadata_marker = (
        "            '    manifest[\"collection\"]"
        "[\"physical_nuisance\"] = dict(nuisance)\\n'\n"
    )
    metadata_replacement = metadata_marker + (
        "            '    manifest[\"collection\"]"
        "[\"post_failure_diagnostic_horizon\"] = {\\n'\n"
        "            '        \"enabled\": True,\\n'\n"
        "            '        \"simulation_advanced\": True,\\n'\n"
        "            '        \"terminal_label_latched\": True,\\n'\n"
        "            '        \"terminal_command\": \"zero_body_velocity\",\\n'\n"
        "            '        \"control_steps_after_first_fall\": 25,\\n'\n"
        "            '        \"minimum_proprio_samples\": 25,\\n'\n"
        "            '        \"minimum_rgb_frames_per_view\": 5,\\n'\n"
        "            '    }\\n'\n"
    )
    if source.count(metadata_marker) != 1:
        raise RuntimeError("v3 provenance insertion point is not unique")
    source = source.replace(metadata_marker, metadata_replacement)

    module = types.ModuleType("kinofail_confirmatory_pair_v3_wrapper")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    exec(compile(source, str(V1_WRAPPER), "exec"), module.__dict__)
    return module


def main() -> None:
    _load_v3_wrapper().main()


if __name__ == "__main__":
    main()
