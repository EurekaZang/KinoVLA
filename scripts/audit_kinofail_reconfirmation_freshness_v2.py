#!/usr/bin/env python3
"""Freshness auditor for the fully preregistered reconfirmation extension.

The implementation is mechanically derived from the hash-pinned v1 auditor.
Only the formal namespace, fresh seed namespaces, material split label, and
the invalid-pilot exposure ledger are changed.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/audit_kinofail_confirmatory_freshness_v1.py"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "ed122f711eab10223420d86a63b0322f6b683b19c19b781758875887c70b6140"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_IMPLEMENTATION_SHA256:
        raise RuntimeError("v1 freshness implementation hash mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        "2_030_000_000": "2_060_000_000",
        "2_040_000_000": "2_070_000_000",
        "confirm_v1_": "confirm_v2_",
        'row.get("split") == "confirmatory"': (
            'row.get("split") == "revised_confirmatory"'
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"freshness patch point absent: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_reconfirmation_freshness_v2_impl")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    module.DEFAULT_PRIOR_EXPOSURES = tuple(module.DEFAULT_PRIOR_EXPOSURES) + (
        "configs/data/kinofail_reconfirmation_exposure_exclusions_v2.json",
    )
    return module


_IMPL = _load()
DEFAULT_PRIOR_EXPOSURES = _IMPL.DEFAULT_PRIOR_EXPOSURES
sha256_file = _IMPL.sha256_file
validate_f0 = _IMPL.validate_f0
_collect_exposure = _IMPL._collect_exposure
audit_freshness = _IMPL.audit_freshness


def main() -> int:
    return int(_IMPL.main())


if __name__ == "__main__":
    raise SystemExit(main())
