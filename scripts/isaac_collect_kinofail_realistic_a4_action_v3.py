#!/usr/bin/env python3
"""Scale A4 with a conclusion-relevant numerical paired-prefix gate.

V2 established that the only strict-hash mismatch was O5 GPU noise (all physical
state differences < 1e-4). V3 keeps the O5 trigger correction and hashes the shared
physical prefix after 1e-3 quantization, while raw rows remain stored for audit.
"""

from __future__ import annotations

import hashlib
import json

from scripts import isaac_collect_kinofail_realistic_a4_action_v2 as v2


implementation = v2.implementation


def _quantize(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_quantize(item) for item in value]
    if isinstance(value, dict):
        return {key: _quantize(item) for key, item in value.items()}
    return value


def _prefix_hash(rows, decision_step):
    digest = hashlib.sha256()
    for row in rows:
        if int(row["step"]) > decision_step:
            break
        compact = {
            key: row[key]
            for key in (
                "step", "time_s", "progress_m", "lateral_m", "position_xy_m",
                "heading_rad", "velocity_body_mps", "base_height_m", "tilt_rad",
                "slip_ratio", "effort_ratio", "support_ratio", "fallen", "command_body",
            )
        }
        digest.update(
            json.dumps(_quantize(compact), sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


implementation._prefix_hash = _prefix_hash


if __name__ == "__main__":
    implementation.main()
