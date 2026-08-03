from __future__ import annotations

import hashlib

from scripts import finalize_kinofail_reconfirmation_v4 as finalizer
from scripts import recover_kinofail_reconfirmation_f14_audit_cache_v1 as recovery


def test_f18_recovered_audit_matches_pruned_inventory_byte_for_byte() -> None:
    canonical, expected, transcript_record_sha256 = (
        recovery.recover_canonical()
    )
    assert len(canonical) == expected["bytes"] == 16279
    assert hashlib.sha256(canonical).hexdigest() == expected["sha256"]
    assert transcript_record_sha256


def test_f18_cache_and_manifest_validate() -> None:
    amendment = finalizer._validate_f18()
    canonical, _, _ = recovery.recover_canonical()
    assert recovery.CACHE.read_bytes() == canonical
    assert amendment["passed"] is True
    assert (
        amendment["correction"]["finalizer_scientific_steps_changed"]
        is False
    )
