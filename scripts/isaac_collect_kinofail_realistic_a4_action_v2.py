#!/usr/bin/env python3
"""One-line protocol correction over the hash-preserved A4-v1 collector.

The v1 preflight showed that severe O5 fell before its 2.2 s branch, so neither
actual action could be evaluated. V2 keeps every scene, seed, operator, action and
outcome rule fixed, but branches O5 after 0.5 s of an already attached payload.
"""

from __future__ import annotations

from scripts import isaac_collect_kinofail_realistic_a4_action_v1 as implementation


_v1_decision_ready = implementation._decision_ready


def _decision_ready(operator, telemetry, engaged_dwell, time_s):
    if operator == "O5_payload":
        return time_s >= 0.5 and implementation._operator_engaged(operator, telemetry)
    return _v1_decision_ready(operator, telemetry, engaged_dwell, time_s)


implementation._decision_ready = _decision_ready


if __name__ == "__main__":
    implementation.main()
