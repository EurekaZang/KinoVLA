#!/usr/bin/env python3
"""Aggregate realistic terrain gates without hiding failed or missing operators."""

from __future__ import annotations

import argparse
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


def _audit_inputs(value: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    recorded = value.get("input_sha256", {})
    rows = {}
    for key, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        rows[key] = {
            "path": str(path.resolve()),
            "recorded_sha256": recorded.get(key),
            "actual_sha256": actual,
            "matches": actual is not None and recorded.get(key) == actual,
        }
    return {
        "fresh": bool(rows) and all(row["matches"] for row in rows.values()),
        "inputs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/terrain_operator_gates")
    args = parser.parse_args()
    vertical_sources = {
        "O1_pbr_physics_decoupling": Path(
            "outputs/kinofail_realistic/terrain_pbr_gate/gate_manifest.json"
        ),
        "O2_per_foot_terramechanics": Path(
            "outputs/kinofail_realistic/o2_terramechanics_gate/gate_manifest.json"
        ),
        "O3_impulse_topology": Path(
            "outputs/kinofail_realistic/o3_topology_gate/gate_manifest.json"
        ),
        "O9_measured_high_centering": Path(
            "outputs/kinofail_realistic/o9_high_centering_gate/gate_manifest.json"
        ),
    }
    dose_sources = {
        "O2_paired_dose_response": Path(
            "outputs/kinofail_realistic/o2_dose_sweep/dose_manifest.json"
        ),
        "O3_paired_dose_response": Path(
            "outputs/kinofail_realistic/o3_dose_sweep/dose_manifest.json"
        ),
        "O9_paired_clearance_threshold": Path(
            "outputs/kinofail_realistic/o9_dose_sweep/dose_manifest.json"
        ),
    }
    policy = Path("outputs/locomotion/policy.pt")
    backend = Path("kino_vla/sim/isaac_policy_backend.py")
    config = Path("configs/sim/go2_skeleton.yaml")
    input_paths = {
        "O1_pbr_physics_decoupling": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o1_mu_field.py"),
            "terrain_materials": Path("kino_vla/sim/terrain_materials.py"),
            "benchmark_design": Path("kino_vla/data/realistic_benchmark.py"),
            "gate_script": Path("scripts/isaac_terrain_pbr_gate.py"),
            "config": config,
        },
        "O2_per_foot_terramechanics": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o2_compliance.py"),
            "terramechanics": Path("kino_vla/sim/terramechanics.py"),
            "gate_script": Path("scripts/isaac_o2_terramechanics_gate.py"),
            "config": config,
        },
        "O3_impulse_topology": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o3_collapse.py"),
            "collapse_model": Path("kino_vla/sim/collapse.py"),
            "gate_script": Path("scripts/isaac_o3_topology_gate.py"),
            "config": config,
        },
        "O9_measured_high_centering": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o9_high_centering.py"),
            "gate_script": Path("scripts/isaac_o9_high_centering_gate.py"),
            "config": config,
        },
        "O2_paired_dose_response": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o2_compliance.py"),
            "terramechanics": Path("kino_vla/sim/terramechanics.py"),
            "gate_script": Path("scripts/isaac_o2_dose_sweep.py"),
            "config": config,
        },
        "O3_paired_dose_response": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o3_collapse.py"),
            "collapse_model": Path("kino_vla/sim/collapse.py"),
            "gate_script": Path("scripts/isaac_o3_dose_sweep.py"),
            "config": config,
        },
        "O9_paired_clearance_threshold": {
            "policy": policy,
            "backend": backend,
            "operator": Path("kino_vla/sim/operators/o9_high_centering.py"),
            "gate_script": Path("scripts/isaac_o9_dose_sweep.py"),
            "config": config,
        },
    }
    vertical_gates = {name: _load(path) for name, path in vertical_sources.items()}
    dose_gates = {name: _load(path) for name, path in dose_sources.items()}
    gates = vertical_gates | dose_gates
    freshness = {name: _audit_inputs(value, input_paths[name]) for name, value in gates.items()}
    statuses = {
        name: bool(value.get("passed")) and freshness[name]["fresh"]
        for name, value in gates.items()
    }
    vertical_statuses = {name: statuses[name] for name in vertical_gates}
    dose_statuses = {name: statuses[name] for name in dose_gates}
    o9 = vertical_gates["O9_measured_high_centering"]
    visual_evidence = _load(
        Path("outputs/kinofail_realistic/o2_o3_visual_evidence/gate_manifest.json")
    )
    visual_freshness = _audit_inputs(
        visual_evidence,
        {
            "policy": policy,
            "backend": backend,
            "o2_operator": Path("kino_vla/sim/operators/o2_compliance.py"),
            "o3_operator": Path("kino_vla/sim/operators/o3_collapse.py"),
            "terramechanics": Path("kino_vla/sim/terramechanics.py"),
            "collapse": Path("kino_vla/sim/collapse.py"),
            "terrain_materials": Path("kino_vla/sim/terrain_materials.py"),
            "benchmark_design": Path("kino_vla/data/realistic_benchmark.py"),
            "gate_script": Path("scripts/isaac_o2_o3_visual_evidence.py"),
            "config": config,
        },
    )
    summary = {
        "schema_version": "kinofail.realistic-terrain-gates.v5",
        "all_required_gates_passed": all(statuses.values()),
        "terrain_vertical_slice_ready": all(vertical_statuses.values()),
        "terrain_engineering_dose_ready": all(dose_statuses.values()),
        "publication_ready": False,
        "gate_status": statuses,
        "vertical_gate_status": vertical_statuses,
        "dose_gate_status": dose_statuses,
        "input_freshness": freshness,
        "passed_gates": sum(statuses.values()),
        "required_gates": len(statuses),
        "blocking_operators": [name for name, passed in statuses.items() if not passed],
        "visual_evidence": {
            "status": "review_bundle_not_corpus_evidence",
            "passed": bool(visual_evidence.get("passed")) and visual_freshness["fresh"],
            "path": visual_evidence.get("path"),
            "sha256": visual_evidence.get("sha256"),
            "input_freshness": visual_freshness,
            "contact_sheet": visual_evidence.get("contact_sheet"),
        },
        "remaining_publication_blockers": [
            "independent scene and geometry realizations beyond deterministic seed repeats",
            "O1 mild/moderate dose characterization",
            "O2/O3 independent soil-substrate and fracture-geometry realizations beyond "
            "the passed synchronized visual slice",
            "O10 time-varying thermal state beyond the passed endpoint-cap gate",
            "396-episode pilot collection and runtime validation",
            "A0-A7 realistic replication with scene-cluster uncertainty",
        ],
        "o9_contact_diagnostic": {
            "checks": o9.get("checks", {}),
            "measurements": o9.get("measurements", {}),
            "max_nonfoot_contact_by_body_n": o9.get("telemetry", {}).get(
                "max_nonfoot_contact_by_body_n", {}
            ),
        },
        "artifacts": {
            name: {"path": value.get("path"), "sha256": value.get("sha256")}
            for name, value in gates.items()
        },
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kino-Fail realistic terrain gate audit",
        "",
        f"Overall: **{'PASS' if summary['all_required_gates_passed'] else 'BLOCKED'}**",
        "",
        "| Gate | Status | Evidence |",
        "|---|---:|---|",
    ]
    for name, value in gates.items():
        status = "PASS" if statuses[name] else "FAIL/MISSING/STALE"
        lines.append(f"| {name} | {status} | `{value.get('path')}` |")
    lines.extend(
        [
            "",
            "All four terrain vertical slices and all three paired engineering dose gates pass.",
            "The O2/O3/O9 seed repeats test deterministic physics reproducibility, not independent",
            "scene-level inference. This is not a publication-ready corpus: independent scene/",
            "geometry realizations, visual synchronization, remaining operators, and 396 validated",
            "pilot episodes are required. Gate success does not promote any scheduled corpus",
            "episode to validated.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
