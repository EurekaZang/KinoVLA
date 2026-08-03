from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_embodiedgen_o4_independence_v6 import _verified_v6_formal


def _write_case(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    pair = tmp_path / "pair_manifest.json"
    pair.write_text(json.dumps({"protocol_role": "formal"}) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "kinofail.embodiedgen-o4-pair-adjudication.v6",
        "adjudication_role": "formal",
        "passed": True,
        "input_pair_manifest": {
            "path": str(pair.resolve()),
            "sha256": hashlib.sha256(pair.read_bytes()).hexdigest(),
        },
        "artifact_checks": {"pair": True},
        "inherited_nonvisual_checks": {"physical": True},
        "phase_aware_visuals": {"passed": True},
    }
    audit.update(overrides)
    audit_path = tmp_path / "phase_aware_v6_formal_audit.json"
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    return pair, audit_path


def test_accepts_hash_bound_formal_v6_pair(tmp_path: Path) -> None:
    pair, audit = _write_case(tmp_path)
    value, adjudication = _verified_v6_formal(pair, audit)
    assert value["protocol_role"] == "formal"
    assert adjudication["passed"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"passed": False},
        {"artifact_checks": {"pair": False}},
        {"inherited_nonvisual_checks": {"physical": False}},
        {"phase_aware_visuals": {"passed": False}},
    ],
)
def test_rejects_failed_v6_or_inherited_gate(
    tmp_path: Path, override: dict[str, object]
) -> None:
    pair, audit = _write_case(tmp_path, **override)
    with pytest.raises(ValueError):
        _verified_v6_formal(pair, audit)


def test_rejects_pair_changed_after_v6_audit(tmp_path: Path) -> None:
    pair, audit = _write_case(tmp_path)
    pair.write_text(json.dumps({"protocol_role": "formal", "tampered": True}) + "\n")
    with pytest.raises(ValueError, match="pair hash"):
        _verified_v6_formal(pair, audit)
