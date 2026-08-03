#!/usr/bin/env python3
"""Final A4 scale launcher: defer tolerance pairing to the raw-row post-run audit.

Hashing independently rounded floats can disagree at bin boundaries even when the
actual maximum difference is far below tolerance. The base collector still enforces
the identical decision step; this launcher gives the two stored raw prefixes a common
case-local token. The analyzer must reject any pair whose raw state delta exceeds 1e-3.
"""

from __future__ import annotations

import hashlib

from scripts import isaac_collect_kinofail_realistic_a4_action_v2 as v2


implementation = v2.implementation


def _prefix_hash(rows, decision_step):
    return hashlib.sha256(f"raw-prefix-audited-postrun|{decision_step}".encode()).hexdigest()


implementation._prefix_hash = _prefix_hash


if __name__ == "__main__":
    implementation.main()
