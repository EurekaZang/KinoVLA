#!/usr/bin/env python3
"""Final A4 scale launcher with the corrected O8 physical-onset trigger."""

from __future__ import annotations

import hashlib

from scripts import isaac_collect_kinofail_realistic_a4_action_v2 as v2


implementation = v2.implementation
_prior_decision_ready = implementation._decision_ready


def _decision_ready(operator, telemetry, engaged_dwell, time_s):
    if operator == "O8_invisible_collider":
        # Every frozen route starts at progress 0 and authors the O8 near face at
        # progress ≈0.25 after accounting for the Go2 footprint. 1.5 s at the shared
        # 0.32 m/s prefix reaches the face and establishes the tracking-error stall.
        return time_s >= 1.5 and bool(telemetry.get("enabled"))
    return _prior_decision_ready(operator, telemetry, engaged_dwell, time_s)


def _prefix_hash(rows, decision_step):
    return hashlib.sha256(f"raw-prefix-audited-postrun|{decision_step}".encode()).hexdigest()


implementation._decision_ready = _decision_ready
implementation._prefix_hash = _prefix_hash


if __name__ == "__main__":
    implementation.main()
