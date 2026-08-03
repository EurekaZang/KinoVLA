#!/usr/bin/env python3
"""Evaluate the v3 candidate gates on G04/G05 calibration data only."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.eval.realistic_operator_admission_v3 import (
    audit_o1_phase_robust_candidate,
    audit_o2_sinkage_normalized_visual_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1"
INPUTS = {
    "g04": {
        "O1": BASE
        / "forest_hybrid_whipple_metric_g04_dev01/isaac_hybrid_composition_dev_v21/g04_o1_crossed_replication_attempt1_pair_audit.json",
        "O2": BASE
        / "forest_hybrid_whipple_metric_g04_dev01/isaac_hybrid_composition_dev_v21/g04_o2_crossed_replication_attempt1/pair_audit.json",
        "O2_anomaly": BASE
        / "forest_hybrid_whipple_metric_g04_dev01/isaac_hybrid_composition_dev_v21/g04_o2_crossed_replication_attempt1/anomaly/lane_manifest.json",
        "O3": BASE
        / "forest_hybrid_whipple_metric_g04_dev01/isaac_hybrid_composition_dev_v21/g04_o3_crossed_replication_attempt1/pair_audit.json",
    },
    "g05": {
        "O1": BASE
        / "forest_hybrid_hochsal_metric_g05_dev01/isaac_hybrid_composition_dev_v22/g05_o1_crossed_replication_attempt1_pair_audit.json",
        "O2": BASE
        / "forest_hybrid_hochsal_metric_g05_dev01/isaac_hybrid_composition_dev_v22/g05_o2_crossed_replication_attempt1/pair_audit.json",
        "O2_anomaly": BASE
        / "forest_hybrid_hochsal_metric_g05_dev01/isaac_hybrid_composition_dev_v22/g05_o2_crossed_replication_attempt1/anomaly/lane_manifest.json",
        "O3": BASE
        / "forest_hybrid_hochsal_metric_g05_dev01/isaac_hybrid_composition_dev_v22/g05_o3_crossed_replication_attempt1/pair_audit.json",
    },
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    scene_results: dict[str, Any] = {}
    inputs: list[dict[str, str]] = []
    strict_v2_passes = 0
    candidate_v3_passes = 0
    for scene_id, paths in INPUTS.items():
        o1_pair = _json(paths["O1"])
        o2_pair = _json(paths["O2"])
        o2_anomaly = _json(paths["O2_anomaly"])
        o3_pair = _json(paths["O3"])
        o1_v3 = audit_o1_phase_robust_candidate(o1_pair)
        o2_v3 = audit_o2_sinkage_normalized_visual_candidate(o2_pair, o2_anomaly)
        o3_v3 = {
            "operator": "O3",
            "passed": o3_pair.get("passed") is True,
            "contract": "unchanged_v2_event_trigger_contract",
            "calibration_only": True,
            "counts_as_a0_a7_evidence": False,
        }
        strict = {
            "O1": o1_pair.get("passed") is True,
            "O2": o2_pair.get("passed") is True,
            "O3": o3_pair.get("passed") is True,
        }
        candidate = {"O1": o1_v3, "O2": o2_v3, "O3": o3_v3}
        strict_v2_passes += sum(strict.values())
        candidate_v3_passes += sum(result["passed"] for result in candidate.values())
        scene_results[scene_id] = {
            "strict_v2_outcomes_retained": strict,
            "candidate_v3_calibration": candidate,
        }
        for label, path in paths.items():
            inputs.append(
                {"scene_id": scene_id, "artifact": label, "path": str(path), "sha256": _sha256(path)}
            )

    result = {
        "schema_version": "kinofail.forest-crossed-operator-admission-v3-calibration.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": candidate_v3_passes == 6,
        "strict_v2_result": {"passed_pairs": strict_v2_passes, "total_pairs": 6},
        "candidate_v3_calibration_result": {
            "passed_pairs": candidate_v3_passes,
            "total_pairs": 6,
        },
        "scene_results": scene_results,
        "inputs": inputs,
        "evidence_boundary": {
            "g04_g05_used_to_design_candidate_gate": True,
            "g04_g05_cannot_confirm_candidate_v3": True,
            "new_unseen_scene_geometry_batch_required": True,
            "scene_registry_eligible": False,
            "counts_as_a0_a7_evidence": False,
        },
        "interpretation": (
            "The v3 candidate explains both strict-v2 misses without changing operator physics, "
            "but 6/6 on the same calibration data is not validation. Promotion requires a frozen "
            "new-scene confirmation batch and the original strict-v2 4/6 must remain reported."
        ),
    }
    output = (
        BASE
        / "forest_crossed_appearance_geometry_o1_o3_replication_dev_v1/operator_admission_v3_calibration.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"], **result["candidate_v3_calibration_result"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
