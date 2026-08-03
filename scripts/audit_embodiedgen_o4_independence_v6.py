#!/usr/bin/env python3
"""Independently audit held-out O4 pairs admitted by phase-aware v6 adjudication."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_embodiedgen_o4_independence import (
    _pair_signature,
    _require,
    _sha256_file,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _passed_calibration(path: Path) -> dict:
    value = _json(path)
    _require(
        value.get("passed") is True,
        f"calibration manifest did not pass its legacy within-pair gates: {path}",
    )
    return value


def _verified_v6_formal(pair_path: Path, audit_path: Path) -> tuple[dict, dict]:
    pair = _json(pair_path)
    audit = _json(audit_path)
    input_spec = audit.get("input_pair_manifest", {})
    _require(
        audit.get("schema_version") == "kinofail.embodiedgen-o4-pair-adjudication.v6",
        f"formal audit is not phase-aware v6: {audit_path}",
    )
    _require(audit.get("adjudication_role") == "formal", f"v6 audit is not formal: {audit_path}")
    _require(audit.get("passed") is True, f"formal pair did not pass v6: {audit_path}")
    _require(
        Path(input_spec.get("path", "")).resolve() == pair_path.resolve(),
        f"v6 audit does not bind the supplied pair path: {audit_path}",
    )
    _require(
        input_spec.get("sha256") == _sha256_file(pair_path),
        f"v6 audit does not bind the supplied pair hash: {audit_path}",
    )
    _require(
        bool(audit.get("artifact_checks")) and all(audit["artifact_checks"].values()),
        f"v6 audit did not verify all pair artifacts: {audit_path}",
    )
    _require(
        bool(audit.get("inherited_nonvisual_checks"))
        and all(audit["inherited_nonvisual_checks"].values()),
        f"v6 audit did not preserve every nonvisual pair gate: {audit_path}",
    )
    _require(
        audit.get("phase_aware_visuals", {}).get("passed") is True,
        f"phase-aware visual contract failed: {audit_path}",
    )
    return pair, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--formal", type=Path, nargs="+", required=True)
    parser.add_argument("--formal-v6-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _require(
        len(args.formal) == len(args.formal_v6_audit),
        "--formal and --formal-v6-audit must have equal lengths",
    )

    protocol_path = args.protocol.resolve()
    protocol = _json(protocol_path)
    protocol_hash = _sha256_file(protocol_path)
    _require(protocol.get("status") == "frozen", "protocol is not frozen")

    calibration = []
    calibration_scene_ids: set[str] = set()
    calibration_scene_families: set[str] = set()
    seen_physics: dict[str, str] = {}
    seen_rgb: dict[str, str] = {}
    for path in args.calibration:
        resolved = path.resolve()
        value = _passed_calibration(resolved)
        _require(
            value.get("protocol_role") == "calibration",
            f"--calibration is not marked calibration: {resolved}",
        )
        record = _pair_signature(resolved, value)
        calibration.append(record)
        scene_id = str(value["scene_id"])
        scene_family = str(value["scene_family"])
        _require(scene_id not in calibration_scene_ids, f"duplicate calibration scene: {scene_id}")
        _require(
            scene_family not in calibration_scene_families,
            f"duplicate calibration scene family: {scene_family}",
        )
        calibration_scene_ids.add(scene_id)
        calibration_scene_families.add(scene_family)
        physics_signature = str(record["paired_physics_sha256"])
        rgb_signature = str(record["paired_rgb_sha256"])
        _require(physics_signature not in seen_physics, f"duplicate calibration trace: {resolved}")
        seen_physics[physics_signature] = str(resolved)
        seen_rgb.setdefault(rgb_signature, str(resolved))

    expected_calibration = {
        (str(Path(row["manifest"]["path"]).resolve()), str(row["manifest"]["sha256"]))
        for row in protocol["calibration_exclusions"]
    }
    observed_calibration = {
        (str(path.resolve()), _sha256_file(path.resolve())) for path in args.calibration
    }
    _require(
        observed_calibration == expected_calibration,
        "supplied calibration manifests do not exactly match frozen exclusions",
    )

    frozen_hash_checks: dict[str, bool] = {}
    for name, frozen in protocol["frozen_files"].items():
        frozen_path = Path(frozen["path"]).resolve()
        frozen_hash_checks[name] = (
            frozen_path.is_file() and _sha256_file(frozen_path) == frozen["sha256"]
        )
    _require(all(frozen_hash_checks.values()), "one or more protocol-frozen files changed")

    formal = []
    for pair_arg, v6_arg in zip(args.formal, args.formal_v6_audit, strict=True):
        resolved = pair_arg.resolve()
        v6_path = v6_arg.resolve()
        value, v6 = _verified_v6_formal(resolved, v6_path)
        _require(value.get("protocol_role") == "formal", f"pair is not formal: {resolved}")
        formal_protocol = value.get("formal_protocol", {})
        binding_checks = {
            "protocol_path": str(Path(formal_protocol.get("path", "")).resolve())
            == str(protocol_path),
            "protocol_sha256": formal_protocol.get("sha256") == protocol_hash,
            "protocol_id": formal_protocol.get("protocol_id") == protocol["protocol_id"],
            "formal_seed": int(value["seed"]) in set(protocol["formal_episode_seeds"]),
            "scene_id": value["scene_id"] == protocol["scene_id"],
            "scene_family": value["scene_family"] == protocol["scene_family"],
            "source_scene_seed": int(value["source_scene_seed"]) == int(protocol["scene_seed"]),
            "heldout_scene_id": str(value["scene_id"]) not in calibration_scene_ids,
            "heldout_scene_family": str(value["scene_family"])
            not in calibration_scene_families,
            "collector_sha256": value["provenance"]["collector_sha256"]
            == protocol["frozen_files"]["collector"]["sha256"],
            "backend_sha256": value["provenance"]["backend_sha256"]
            == protocol["frozen_files"]["backend"]["sha256"],
            "adhesion_model_sha256": value["provenance"]["adhesion_model_sha256"]
            == protocol["frozen_files"]["adhesion_model"]["sha256"],
            "episode_usd_sha256": value["episode_usd_sha256"]
            == protocol["frozen_files"]["episode_usd"]["sha256"],
            "compiled_audit_sha256": value["compiled_audit_sha256"]
            == protocol["frozen_files"]["compiled_audit"]["sha256"],
            "rtx_audit_sha256": value["rtx_qa_sha256"]
            == protocol["frozen_files"]["rtx_audit"]["sha256"],
            "go2_audit_sha256": value["go2_qa_sha256"]
            == protocol["frozen_files"]["go2_audit"]["sha256"],
        }
        _require(all(binding_checks.values()), f"formal pair violates frozen protocol: {resolved}")
        record = _pair_signature(resolved, value)
        record["pair_manifest_sha256"] = _sha256_file(resolved)
        record["phase_aware_v6_audit"] = {
            "path": str(v6_path),
            "sha256": _sha256_file(v6_path),
            "passed": v6["passed"],
        }
        record["protocol_binding_checks"] = binding_checks
        physics_signature = str(record["paired_physics_sha256"])
        rgb_signature = str(record["paired_rgb_sha256"])
        physics_duplicate_of = seen_physics.get(physics_signature)
        rgb_duplicate_of = seen_rgb.get(rgb_signature)
        record["independent_physics"] = physics_duplicate_of is None
        record["independent_rgb"] = rgb_duplicate_of is None
        record["independent_benchmark_pair"] = physics_duplicate_of is None
        record["physics_duplicate_of"] = physics_duplicate_of
        record["rgb_duplicate_of"] = rgb_duplicate_of
        if physics_duplicate_of is None:
            seen_physics[physics_signature] = str(resolved)
        if rgb_duplicate_of is None:
            seen_rgb[rgb_signature] = str(resolved)
        formal.append(record)

    independent_count = sum(bool(record["independent_benchmark_pair"]) for record in formal)
    passed = independent_count == len(formal)
    audit = {
        "schema_version": "kinofail.embodiedgen-o4-formal-admission-audit.v3-phase-aware",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_hash,
            "protocol_id": protocol["protocol_id"],
            "frozen_file_hash_checks": frozen_hash_checks,
        },
        "calibration_pairs": calibration,
        "calibration_pair_count": len(calibration),
        "calibration_scene_family_count": len(calibration_scene_families),
        "formal_pairs": formal,
        "formal_pair_count": len(formal),
        "independent_formal_pair_count": independent_count,
        "all_formal_pairs_phase_aware_v6_admitted": all(
            record["phase_aware_v6_audit"]["passed"] for record in formal
        ),
        "all_formal_pairs_physics_independent": passed,
        "all_formal_pairs_protocol_bound": all(
            all(record["protocol_binding_checks"].values()) for record in formal
        ),
        "formal_scene_held_out_from_all_calibration_families": all(
            str(record["scene_id"]) not in calibration_scene_ids for record in formal
        ),
        "publication_interpretation": (
            "frozen, phase-aware-v6-admitted, held-out, physically independent formal O4 evidence"
            if passed
            else "duplicate physical trajectories are reproducibility evidence only"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
