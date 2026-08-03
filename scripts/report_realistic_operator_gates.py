#!/usr/bin/env python3
"""Aggregate all implemented Kino-Fail realistic operator engineering gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "missing": True, "path": str(path.resolve())}
    value = json.loads(path.read_text(encoding="utf-8"))
    value["path"] = str(path.resolve())
    value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def _fresh(value: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    recorded = value.get("input_sha256", {})
    mismatches = []
    for key, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual is None or recorded.get(key) != actual:
            mismatches.append(
                {
                    "key": key,
                    "path": str(path.resolve()),
                    "recorded_sha256": recorded.get(key),
                    "actual_sha256": actual,
                }
            )
    return {"fresh": not mismatches, "mismatches": mismatches}


def _terrain_live_status(terrain: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    """Recheck hashes embedded in the terrain aggregate so it cannot itself go stale."""
    statuses: dict[str, bool] = {}
    audits: dict[str, Any] = {}
    for name, passed in terrain.get("gate_status", {}).items():
        mismatches = []
        recorded_audit = terrain.get("input_freshness", {}).get(name, {})
        for key, row in recorded_audit.get("inputs", {}).items():
            path = Path(str(row.get("path", "")))
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            expected = row.get("actual_sha256")
            if actual is None or actual != expected or not bool(row.get("matches")):
                mismatches.append(
                    {
                        "key": key,
                        "path": str(path),
                        "terrain_summary_sha256": expected,
                        "actual_sha256": actual,
                    }
                )
        fresh = bool(recorded_audit.get("inputs")) and not mismatches
        statuses[name] = bool(passed) and fresh
        audits[name] = {"fresh": fresh, "mismatches": mismatches}
    return statuses, audits


def main() -> None:
    output = Path("outputs/kinofail_realistic/operator_gates")
    output.mkdir(parents=True, exist_ok=True)
    terrain = _load(Path("outputs/kinofail_realistic/terrain_operator_gates/summary.json"))
    gate_paths = {
        "O4_foot_local_adhesion_dose": Path(
            "outputs/kinofail_realistic/o4_adhesion_dose_sweep/dose_manifest.json"
        ),
        "O5_rigid_payload_vertical_slice": Path(
            "outputs/kinofail_realistic/o5_payload_gate/gate_manifest.json"
        ),
        "O5_paired_mass_dose": Path("outputs/kinofail_realistic/o5_dose_sweep/dose_manifest.json"),
        "O6_finite_wrench_dose": Path(
            "outputs/kinofail_realistic/o6_force_pulse_dose_sweep/dose_manifest.json"
        ),
        "O7_go2_front_rtx_depth_contract": Path(
            "outputs/kinofail_realistic/o7_rtx_depth_gate/gate_manifest.json"
        ),
        "O8_transparent_obstacle_contract": Path(
            "outputs/kinofail_realistic/o8_transparent_gate/gate_manifest.json"
        ),
        "O10_paired_actuator_cap_dose": Path(
            "outputs/kinofail_realistic/o10_dose_sweep/dose_manifest.json"
        ),
        "O11_raw_proprio_fault_dose": Path(
            "outputs/kinofail_realistic/o11_sensor_fault_gate/gate_manifest.json"
        ),
    }
    gates = {name: _load(path) for name, path in gate_paths.items()}
    common = {
        "policy": Path("outputs/locomotion/policy.pt"),
        "backend": Path("kino_vla/sim/isaac_policy_backend.py"),
        "operator": Path("kino_vla/sim/operators/o5_payload.py"),
        "mass_model": Path("kino_vla/sim/payload.py"),
        "actuator_model": Path("kino_vla/sim/actuator.py"),
        "config": Path("configs/sim/go2_skeleton.yaml"),
    }
    source_paths = {
        "O4_foot_local_adhesion_dose": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator_model": Path("kino_vla/sim/adhesion.py"),
            "scene_compiler": Path("kino_vla/sim/realistic_scene.py"),
            "gate_script": Path("scripts/isaac_o4_adhesion_dose_sweep.py"),
            "sim_config": common["config"],
            "demo_config": Path("configs/demo/indoor_adhesion_icra.yaml"),
        },
        "O5_rigid_payload_vertical_slice": {
            key: path for key, path in common.items() if key != "actuator_model"
        }
        | {"gate_script": Path("scripts/isaac_o5_payload_gate.py")},
        "O5_paired_mass_dose": common | {"gate_script": Path("scripts/isaac_o5_dose_sweep.py")},
        "O6_finite_wrench_dose": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator": Path("kino_vla/sim/operators/o6_push.py"),
            "gate_script": Path("scripts/isaac_o6_force_pulse_dose_sweep.py"),
            "config": common["config"],
        },
        "O7_go2_front_rtx_depth_contract": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator": Path("kino_vla/sim/operators/o7_visual_remap.py"),
            "depth_pipeline": Path("kino_vla/sim/depth_pipeline.py"),
            "live_segmenter": Path("kino_vla/map/live_rtx_segmenter.py"),
            "terrain_materials": Path("kino_vla/sim/terrain_materials.py"),
            "benchmark_design": Path("kino_vla/data/realistic_benchmark.py"),
            "asset_lock": Path("outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"),
            "gate_script": Path("scripts/isaac_o7_rtx_depth_gate.py"),
            "config": common["config"],
        },
        "O8_transparent_obstacle_contract": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator": Path("kino_vla/sim/operators/o8_invisible_collider.py"),
            "types": Path("kino_vla/sim/types.py"),
            "gate_script": Path("scripts/isaac_o8_transparent_obstacle_gate.py"),
            "config": common["config"],
        },
        "O10_paired_actuator_cap_dose": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator": Path("kino_vla/sim/operators/o10_effort_decay.py"),
            "actuator_model": common["actuator_model"],
            "gate_script": Path("scripts/isaac_o10_dose_sweep.py"),
            "config": common["config"],
        },
        "O11_raw_proprio_fault_dose": {
            "policy": common["policy"],
            "backend": common["backend"],
            "operator": Path("kino_vla/sim/operators/o11_obs_bias.py"),
            "sensor_pipeline": Path("kino_vla/sim/proprio_pipeline.py"),
            "gate_script": Path("scripts/isaac_o11_sensor_fault_gate.py"),
            "config": common["config"],
        },
    }
    freshness = {name: _fresh(gates[name], source_paths[name]) for name in gates}
    statuses = {
        name: bool(value.get("passed")) and freshness[name]["fresh"]
        for name, value in gates.items()
    }
    terrain_statuses, terrain_freshness = _terrain_live_status(terrain)
    terrain_passed = bool(terrain_statuses) and all(terrain_statuses.values())
    terrain_count = sum(terrain_statuses.values())
    implemented_count = int(terrain.get("required_gates", 7)) + len(gates)
    passed_count = terrain_count + sum(statuses.values())
    adhesion_demo = _load(Path("outputs/demos/indoor_adhesion_icra/manifest.json"))
    terrain_visuals = _load(
        Path("outputs/kinofail_realistic/o2_o3_visual_evidence/gate_manifest.json")
    )
    terrain_visual_freshness = _fresh(
        terrain_visuals,
        {
            "policy": common["policy"],
            "backend": common["backend"],
            "o2_operator": Path("kino_vla/sim/operators/o2_compliance.py"),
            "o3_operator": Path("kino_vla/sim/operators/o3_collapse.py"),
            "terramechanics": Path("kino_vla/sim/terramechanics.py"),
            "collapse": Path("kino_vla/sim/collapse.py"),
            "terrain_materials": Path("kino_vla/sim/terrain_materials.py"),
            "benchmark_design": Path("kino_vla/data/realistic_benchmark.py"),
            "gate_script": Path("scripts/isaac_o2_o3_visual_evidence.py"),
            "config": common["config"],
        },
    )
    runtime_formal = _load(
        Path("outputs/kinofail_realistic/corpus_v1_formal/o4_formal_pilot_summary.json")
    )
    formal_protocol = _load(Path("configs/data/kinofail_o4_formal_pilot_v1.json"))
    protocol_mismatches = []
    for relative, expected in formal_protocol.get("collector_components", {}).items():
        component = Path(relative)
        actual = hashlib.sha256(component.read_bytes()).hexdigest() if component.is_file() else None
        if actual != expected:
            protocol_mismatches.append(
                {"path": str(component.resolve()), "expected_sha256": expected, "actual_sha256": actual}
            )
    for key, path in (
        ("runtime_manifest_sha256", Path("kino_vla/data/runtime_manifest.py")),
        ("schedule_sha256", Path("outputs/kinofail_realistic/design_v1/pilot_schedule.jsonl")),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != formal_protocol.get(key):
            protocol_mismatches.append(
                {
                    "path": str(path.resolve()),
                    "expected_sha256": formal_protocol.get(key),
                    "actual_sha256": actual,
                }
            )
    formal_protocol_freshness = {
        "fresh": not protocol_mismatches
        and runtime_formal.get("protocol_sha256") == formal_protocol.get("sha256")
        and runtime_formal.get("collector_bundle_sha256")
        == formal_protocol.get("collector_sha256"),
        "mismatches": protocol_mismatches,
    }
    runtime_audit = _load(
        Path("outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/runtime_audit.json")
    )
    runtime_coverage = _load(
        Path("outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/coverage.json")
    )
    pilot_schedule = Path("outputs/kinofail_realistic/design_v1/pilot_schedule.jsonl")
    runtime_config = Path("configs/data/kinofail_realistic.yaml")
    eligible_manifest = Path(
        "outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/evaluation_eligible.jsonl"
    )
    live_schedule_sha = hashlib.sha256(pilot_schedule.read_bytes()).hexdigest()
    live_runtime_config_sha = hashlib.sha256(runtime_config.read_bytes()).hexdigest()
    live_eligible_sha = hashlib.sha256(eligible_manifest.read_bytes()).hexdigest()
    runtime_schedule_fresh = runtime_formal.get("schedule_sha256") == live_schedule_sha
    runtime_audit_fresh = (
        runtime_audit.get("schedule_sha256") == live_schedule_sha
        and runtime_audit.get("runtime_config_sha256") == live_runtime_config_sha
        and runtime_audit.get("eligible_manifest_sha256") == live_eligible_sha
    )
    runtime_formal_passed = (
        bool(runtime_formal.get("passed"))
        and formal_protocol_freshness["fresh"]
        and runtime_schedule_fresh
        and runtime_audit_fresh
        and len(runtime_formal.get("results", [])) == 4
        and all(bool(row.get("passed")) for row in runtime_formal.get("results", []))
        and runtime_audit.get("evaluation_eligible_records") == 4
        and runtime_audit.get("complete_counterfactual_pairs") == 2
        and runtime_coverage.get("measured_texture_swap", {}).get("all_pass") is True
    )
    summary = {
        "schema_version": "kinofail.realistic-operator-gates.v11",
        "all_implemented_engineering_gates_passed": terrain_passed and all(statuses.values()),
        "publication_ready": False,
        "passed_engineering_gates": passed_count,
        "implemented_engineering_gates": implemented_count,
        "terrain_gate_summary": {
            "path": terrain.get("path"),
            "sha256": terrain.get("sha256"),
            "passed": terrain_passed,
            "passed_gates": terrain.get("passed_gates"),
            "required_gates": terrain.get("required_gates"),
            "live_gate_status": terrain_statuses,
            "live_input_freshness": terrain_freshness,
        },
        "nonterrain_gate_status": statuses,
        "nonterrain_input_freshness": freshness,
        "adhesion_status": {
            "status": "demo_only_not_engineering_gate",
            "path": adhesion_demo.get("path"),
            "paper_status": adhesion_demo.get("paper_status"),
        },
        "terrain_visual_evidence": {
            "status": "review_bundle_not_corpus_evidence",
            "passed": bool(terrain_visuals.get("passed")) and terrain_visual_freshness["fresh"],
            "path": terrain_visuals.get("path"),
            "sha256": terrain_visuals.get("sha256"),
            "input_freshness": terrain_visual_freshness,
            "contact_sheet": terrain_visuals.get("contact_sheet"),
        },
        "runtime_vertical_slice": {
            "status": "formal_partial_evaluation_eligible_not_publication_freeze",
            "artifact_qa_passed": runtime_formal_passed,
            "evaluation_eligible": runtime_formal_passed,
            "formal_summary_path": runtime_formal.get("path"),
            "formal_summary_sha256": runtime_formal.get("sha256"),
            "protocol_path": formal_protocol.get("path"),
            "protocol_sha256": formal_protocol.get("sha256"),
            "input_freshness": formal_protocol_freshness,
            "schedule_fresh": runtime_schedule_fresh,
            "runtime_audit_fresh": runtime_audit_fresh,
            "runtime_audit_path": runtime_audit.get("path"),
            "runtime_audit_sha256": runtime_audit.get("sha256"),
            "coverage_path": runtime_coverage.get("path"),
            "coverage_sha256": runtime_coverage.get("sha256"),
            "scheduled_records": runtime_audit.get("scheduled_records"),
            "validated_smoke_records": runtime_audit.get("validated_smoke_records"),
            "evaluation_eligible_records": runtime_audit.get("evaluation_eligible_records"),
            "complete_counterfactual_pairs": runtime_audit.get("complete_counterfactual_pairs"),
            "publication_freeze_ready": runtime_audit.get("publication_freeze_ready"),
        },
        "remaining_publication_blockers": [
            "O2/O3 independent soil-substrate and fracture-geometry realizations beyond "
            "the passed synchronized visual slice",
            "O5 independent payload appearance and mount-pose realizations",
            "O5 task-recovery consequence gate beyond descriptive actuator telemetry",
            "O4 independent indoor scene and adhesion-appearance realizations",
            "formal texture-swap model consistency audit on A0-A7 predictions",
            "O10 time-varying thermal state/ramp beyond endpoint-cap characterization",
            "independent scene and geometry realizations beyond deterministic seed repeats",
            "remaining 392/396 pilot episodes and 196/198 counterfactual pairs",
            "A0-A7 realistic replication with scene-cluster uncertainty",
            "real Go2 fixture calibration and blinded anchor evaluation",
        ],
        "artifacts": {
            name: {"path": value.get("path"), "sha256": value.get("sha256")}
            for name, value in gates.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kino-Fail realistic operator engineering audit",
        "",
        f"Implemented gates: **{passed_count}/{implemented_count} PASS**",
        "",
        "| Evidence group | Status |",
        "|---|---:|",
        f"| O1/O2/O3/O9 terrain gates | {'PASS' if terrain_passed else 'FAIL/STALE'} |",
    ]
    for name, passed in statuses.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL/STALE'} |")
    lines.extend(
        [
            "",
            "O4's formal foot-local dose gate is included above. Its rendered indoor bundle is",
            "kept separately as visual demo evidence, not counted as an additional gate. The",
            "benchmark remains not publication-ready until the",
            "listed operator, corpus, A0-A7 replication, and real-fixture blockers are closed.",
            "",
            "The first frozen O4 formal subset contains four evaluation-eligible physical",
            "episodes (two complete pairs, twelve synchronized appearance sequences). This is",
            "a partial pilot result: 392/396 episodes and the remaining scene/operator coverage",
            "are still required before publication freeze.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
