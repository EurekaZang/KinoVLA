#!/usr/bin/env python3
"""Freeze F32c, a fresh excluded pilot for the 0.18 m/s O9 approach."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import build_kinofail_confirmatory_o9_pilot_f32b as base


ROOT = base.ROOT
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32c"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_confirmatory_o9_pilot_f32c/corpus")
RUNNER = ROOT / "scripts/run_kinofail_confirmatory_o9_pilot_f32c.py"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v2.py"
F32B_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32b/final_audit.json"
SEED_BASE = 2_332_000_000


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F32c pilot or corpus")
    f32b = json.loads(F32B_AUDIT.read_text())
    semantic_passes = sum(
        bool(row.get("evidence", {}).get("anomaly_semantic", {}).get("passed"))
        for row in f32b.get("results", [])
    )
    if (
        f32b.get("status") != "final"
        or f32b.get("passed") is not False
        or f32b.get("model_prediction_feature_label_or_score_read") is not False
        or f32b.get("counts", {}).get("scheduled_pairs") != 12
        or semantic_passes != 12
    ):
        raise RuntimeError("F32b audit does not justify a speed-only excluded pilot revision")

    original_clone = base.clone_pair

    def clone_pair(source: list[dict], index: int) -> list[dict]:
        rows = original_clone(source, index)
        old_id = str(rows[0]["counterfactual_group_id"])
        new_id = base.base.make_id(
            "cf_f32c_",
            {
                "source": source[0]["counterfactual_group_id"],
                "seed": SEED_BASE + index,
                "pilot_attempt": 3,
                "speed_revision_mps": 0.18,
            },
        )
        rows = base.base.replace_token(rows, old_id, new_id)
        for row in rows:
            condition = str(row["condition"])
            row.update(
                {
                    "benchmark_id": "kinofail_confirmatory_o9_pilot_f32c",
                    "counterfactual_group_id": new_id,
                    "episode_id": f"{new_id}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                    "design_mode": "excluded_physical_mechanism_pilot_speed_revision",
                    "split": "excluded_o9_mechanism_pilot_f32c",
                }
            )
            row["physical_nuisance"]["forward_speed_mps"] = 0.18
            row["o9_semantic_recollection"].update(
                {
                    "phase": "excluded_pilot_speed_revision_f32c",
                    "pilot_pair_or_seed_reused": False,
                    "adaptation_basis": "F32b retained direct semantics but 0.08 m/s caused nominal route drift",
                }
            )
        return rows

    base.OUTPUT = OUTPUT
    base.CORPUS = CORPUS
    base.RUNNER = RUNNER
    base.F32_AUDIT = F32B_AUDIT
    base.SEED_BASE = SEED_BASE
    base.clone_pair = clone_pair
    base.base.COLLECTOR = COLLECTOR
    result = base.main()

    design_path = OUTPUT / "design.json"
    design = json.loads(design_path.read_text())
    design.update(
        {
            "schema_version": "kinofail.confirmatory-o9-pilot-f32c-design.v1",
            "status": "fixed_before_third_pilot_collection",
            "scientific_role": "excluded model-blind speed-stability and direct-mechanism pilot",
            "f32b_observation": design.pop("f32_observation"),
            "single_prespecified_change": {
                "forward_speed_mps": [0.08, 0.18],
                "unchanged_start_lateral_offset_m": [-0.003, 0.003],
                "unchanged_start_heading_offset_rad": [-0.002, 0.002],
                "unchanged_start_progress_m": 0.35,
                "unchanged_spawn_base_height_m": 0.43,
                "unchanged_o9_geometry_and_semantic_thresholds": True,
            },
        }
    )
    base.base.write_json(design_path, design)

    protocol_path = OUTPUT / "collection_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        {
            "protocol_variant": "f32c-locomotion-stable-approach-speed",
            "benchmark_id": "kinofail_confirmatory_o9_pilot_f32c",
            "collector_path": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": base.base.sha256(COLLECTOR),
            "design_sha256": base.base.sha256(design_path),
        }
    )
    base.base.write_json(protocol_path, protocol)

    seal_path = OUTPUT / "seal_manifest.json"
    seal = json.loads(seal_path.read_text())
    seal.update(
        {
            "schema_version": "kinofail.confirmatory-o9-pilot-f32c-seal.v1",
            "design_sha256": base.base.sha256(design_path),
            "protocol_sha256": base.base.sha256(protocol_path),
            "f32b_audit": seal.pop("f32_audit"),
            "f32b_audit_sha256": seal.pop("f32_audit_sha256"),
        }
    )
    base.base.write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(
        f"{base.base.sha256(seal_path)}  {seal_path.name}\n"
    )
    print(json.dumps(seal, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
