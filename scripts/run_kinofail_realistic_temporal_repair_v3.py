#!/usr/bin/env python3
"""Collect and verify the 90-pair complete O3/O8 severe repair overlay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/run_kinofail_realistic_o3_temporal_repair_v1.py"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_temporal_repair_formal_v3.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_temporal_repair_v3.py"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v3"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_base():
    spec = importlib.util.spec_from_file_location("kinofail_temporal_repair_runner_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accepted(module, pair_id: str, protocol_id: str) -> tuple[bool, list[str]]:
    from kino_vla.data.realistic_snapshots import _runtime_is_accepted

    rows = [
        json.loads(line)
        for line in module.SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    schedule_by_episode = {str(row["episode_id"]): row for row in rows}
    summary = _json(CORPUS / "pair_summaries" / f"{pair_id}.json")
    results = summary.get("results")
    issues: list[str] = []
    if not isinstance(results, list) or len(results) != 2:
        return False, ["missing_or_incomplete_pair_summary"]
    conditions: set[str] = set()
    nuisances: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            issues.append("invalid_pair_result")
            continue
        condition = str(result.get("condition"))
        conditions.add(condition)
        record = schedule_by_episode.get(str(result.get("episode_id")))
        manifest = _json(Path(str(result.get("manifest", ""))))
        if record is None or not manifest:
            issues.append(f"missing_schedule_or_manifest:{condition}")
            continue
        if (
            record.get("severity_id") != "severe"
            or record.get("target_operator")
            not in {"O3_collapse", "O8_invisible_collider"}
        ):
            issues.append(f"episode_outside_repair_scope:{result.get('episode_id')}")
        if (
            manifest.get("collection", {})
            .get("formal_protocol", {})
            .get("protocol_id")
            != protocol_id
        ):
            issues.append(f"repair_protocol_mismatch:{condition}")
        repair = manifest.get("collection", {}).get("temporal_window_repair", {})
        if (
            repair.get("pair_shared_pre_operator_steps") != 30
            or repair.get("pre_operator_duration_s") != 0.6
        ):
            issues.append(f"repair_readback_mismatch:{condition}")
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if not isinstance(nuisance, dict):
            issues.append(f"physical_nuisance_missing:{condition}")
        else:
            nuisances.append(nuisance)
        if not _runtime_is_accepted(
            record,
            manifest,
            allowed_suffixes=[
                "appearance_effect_too_small",
                "rgb_spatial_contrast_too_low",
            ],
        ):
            issues.extend(
                str(value)
                for value in manifest.get("runtime_validation", {}).get("issues", [])
            )
    if conditions != {"anomaly", "nominal_counterfactual"}:
        issues.append("condition_pair_incomplete")
    if len(nuisances) == 2 and (
        nuisances[0] != nuisances[1]
        or nuisances[0].get("profile_index") not in range(5)
        or nuisances[0].get("pair_shared") is not True
    ):
        issues.append("repair_pair_nuisance_mismatch")
    return not issues, sorted(set(issues))


def main() -> int:
    module = _load_base()
    module.PROTOCOL = PROTOCOL
    module.COLLECTOR = COLLECTOR
    module.CORPUS = CORPUS
    module._accepted = lambda pair_id, protocol_id: _accepted(
        module, pair_id, protocol_id
    )
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
