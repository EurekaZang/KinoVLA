from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/repair_kinofail_realistic_scale_v8_structural_v1.py"
SPEC = importlib.util.spec_from_file_location("scale_v8_structural_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _pair(tmp_path: Path, *, shared: bool = True, summary_passed: bool = False) -> tuple[Path, str]:
    pair_id = "cf_test"
    results = []
    nuisances = [
        {"profile_index": 2, "pair_shared": True},
        {"profile_index": 2 if shared else 3, "pair_shared": True},
    ]
    for condition, nuisance in zip(
        ("nominal_counterfactual", "anomaly"), nuisances, strict=True
    ):
        manifest = tmp_path / condition / "manifest.json"
        _write(
            manifest,
            {
                "collection": {
                    "formal_protocol": {"protocol_id": MODULE.PROTOCOL_ID},
                    "physical_nuisance": nuisance,
                }
            },
        )
        results.append({"condition": condition, "manifest": str(manifest)})
    _write(
        tmp_path / "pair_summaries" / f"{pair_id}.json",
        {"passed": summary_passed, "results": results},
    )
    return tmp_path, pair_id


def test_complete_negative_qa_pair_is_not_retried(tmp_path: Path) -> None:
    corpus, pair_id = _pair(tmp_path, summary_passed=False)
    complete, issues = MODULE._structurally_complete(corpus, pair_id)
    assert complete is True
    assert issues == []


def test_pair_shared_nuisance_mismatch_is_structural_failure(tmp_path: Path) -> None:
    corpus, pair_id = _pair(tmp_path, shared=False)
    complete, issues = MODULE._structurally_complete(corpus, pair_id)
    assert complete is False
    assert "pair_shared_physical_nuisance_mismatch" in issues


def test_missing_summary_is_structural_failure(tmp_path: Path) -> None:
    complete, issues = MODULE._structurally_complete(tmp_path, "cf_missing")
    assert complete is False
    assert issues == ["missing_or_incomplete_pair_summary"]
