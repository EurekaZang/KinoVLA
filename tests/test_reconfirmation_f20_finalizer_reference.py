from __future__ import annotations

from scripts import finalize_kinofail_reconfirmation_v4 as predecessor
from scripts import finalize_kinofail_reconfirmation_v6 as finalizer


def test_f20_resolves_the_frozen_v2_finalizer() -> None:
    assert finalizer.BASE_FINALIZER is predecessor.predecessor.base
    assert hasattr(finalizer.BASE_FINALIZER, "EVAL_ROOT")
    assert hasattr(finalizer.BASE_FINALIZER, "_wait")


def test_f20_reuses_the_frozen_v4_finalizer_main() -> None:
    assert finalizer.predecessor.main is predecessor.main


def test_f20_f19_recovery_still_validates() -> None:
    amendment = finalizer._validate_recovery()
    assert amendment["passed"] is True


def test_f20_manifest_validates_after_seal() -> None:
    amendment = finalizer._validate_f20()
    assert amendment["passed"] is True
    assert amendment["correction"]["module_reference_only_changed"] is True
