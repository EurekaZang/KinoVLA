from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_kinofail_realistic_icra_readiness_v1.py"
SPEC = importlib.util.spec_from_file_location("icra_readiness_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _summary(root: Path, pair_id: str, *, accepted: bool) -> None:
    results = []
    for condition in ("nominal_counterfactual", "anomaly"):
        episode_id = f"{pair_id}_{condition}"
        manifest = root / "episodes" / episode_id / "manifest.json"
        _write(
            manifest,
            {
                "runtime_validation": {
                    "passed": accepted,
                    "issues": [] if accepted else ["too_few_proprio_samples"],
                }
            },
        )
        results.append(
            {
                "episode_id": episode_id,
                "condition": condition,
                "manifest": str(manifest),
            }
        )
    _write(
        root / "pair_summaries" / f"{pair_id}.json",
        {"counterfactual_group_id": pair_id, "results": results},
    )


def _schedule(pair_id: str) -> list[dict]:
    return [
        {
            "episode_id": f"{pair_id}_{condition}",
            "counterfactual_group_id": pair_id,
            "scene_family": "scene_a",
            "scene_seed": 11,
            "target_operator": "O3_collapse",
        }
        for condition in ("nominal_counterfactual", "anomaly")
    ]


def test_repair_overlay_replaces_rejected_base_pair(tmp_path: Path) -> None:
    base, repair = tmp_path / "base", tmp_path / "repair"
    pair_id = "cf_o3"
    _summary(base, pair_id, accepted=False)
    _summary(repair, pair_id, accepted=True)
    progress = MODULE._runtime_progress(
        _schedule(pair_id),
        base,
        [],
        repair_corpus=repair,
        repair_protocol={"allowed": {"counterfactual_group_ids": [pair_id]}},
    )
    assert progress["finished_pairs"] == 1
    assert progress["scientifically_eligible_finished_pairs"] == 1
    assert progress["scientifically_rejected_finished_pairs"] == 0
    assert progress["repair_overlay_pairs_used"] == [pair_id]


def test_missing_repair_falls_back_to_base_pair(tmp_path: Path) -> None:
    base, repair = tmp_path / "base", tmp_path / "repair"
    pair_id = "cf_o3"
    _summary(base, pair_id, accepted=False)
    progress = MODULE._runtime_progress(
        _schedule(pair_id),
        base,
        [],
        repair_corpus=repair,
        repair_protocol={"allowed": {"counterfactual_group_ids": [pair_id]}},
    )
    assert progress["scientifically_eligible_finished_pairs"] == 0
    assert progress["scientifically_rejected_finished_pairs"] == 1
    assert progress["repair_overlay_pair_count"] == 0
