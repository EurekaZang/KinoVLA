#!/usr/bin/env python3
"""Audit held-out O4 pairs against every calibration and the frozen protocol.

The collector's ``passed`` bit is necessary but not sufficient for formal admission.  This
auditor independently verifies protocol binding, frozen-file hashes, scene-family exclusion,
and physical-trace uniqueness across all calibration and formal pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("passed"):
        raise ValueError(f"pair manifest did not pass its within-pair gates: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ordered_rgb_signature(pair_root: Path, relative_manifest: str) -> str:
    records = [
        json.loads(line)
        for line in (pair_root / relative_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hashes = [str(record["sha256"]) for record in records]
    return _sha256_bytes("\n".join(hashes).encode("ascii"))


def _pair_signature(path: Path, value: dict[str, object]) -> dict[str, object]:
    takes = value["takes"]
    signatures: dict[str, object] = {}
    for take in ("nominal", "o4_adhesion"):
        summary = takes[take]
        signatures[take] = {
            "telemetry_sha256": summary["telemetry_sha256"],
            "ordered_rgb_content_sha256": _ordered_rgb_signature(
                path.parent, str(summary["frames_manifest"])
            ),
        }
    return {
        "manifest": str(path.resolve()),
        "protocol_role": value["protocol_role"],
        "scene_id": value["scene_id"],
        "seed": value["seed"],
        "takes": signatures,
        "paired_physics_sha256": _sha256_bytes(
            "\n".join(
                str(signatures[take]["telemetry_sha256"])
                for take in ("nominal", "o4_adhesion")
            ).encode("ascii")
        ),
        "paired_rgb_sha256": _sha256_bytes(
            "\n".join(
                str(signatures[take]["ordered_rgb_content_sha256"])
                for take in ("nominal", "o4_adhesion")
            ).encode("ascii")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--formal", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = _sha256_file(protocol_path)
    _require(protocol.get("status") == "frozen", "protocol is not frozen")

    calibration = []
    calibration_scene_ids: set[str] = set()
    calibration_scene_families: set[str] = set()
    seen_physics: dict[str, str] = {}
    seen_rgb: dict[str, str] = {}
    for path in args.calibration:
        resolved = path.resolve()
        value = _read_manifest(resolved)
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
        _require(
            physics_signature not in seen_physics,
            f"duplicate calibration physical trace: {resolved}",
        )
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
        "supplied calibration manifests do not exactly match the frozen exclusions",
    )

    frozen_hash_checks: dict[str, bool] = {}
    for name, frozen in protocol["frozen_files"].items():
        frozen_path = Path(frozen["path"]).resolve()
        ok = frozen_path.is_file() and _sha256_file(frozen_path) == frozen["sha256"]
        frozen_hash_checks[name] = ok
    _require(all(frozen_hash_checks.values()), "one or more frozen files changed after freeze")

    formal = []
    for path in args.formal:
        resolved = path.resolve()
        value = _read_manifest(resolved)
        _require(
            value.get("protocol_role") == "formal",
            f"formal manifest is not marked formal: {resolved}",
        )
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
        _require(
            all(binding_checks.values()),
            f"formal pair violates frozen protocol: {resolved}; checks={binding_checks}",
        )
        record = _pair_signature(resolved, value)
        record["protocol_binding_checks"] = binding_checks
        physics_signature = str(record["paired_physics_sha256"])
        rgb_signature = str(record["paired_rgb_sha256"])
        physics_duplicate_of = seen_physics.get(physics_signature)
        rgb_duplicate_of = seen_rgb.get(rgb_signature)
        record["independent_physics"] = physics_duplicate_of is None
        record["independent_rgb"] = rgb_duplicate_of is None
        # RTX sampling noise does not turn an identical physical trajectory into an independent
        # attribution experiment.  Formal admission therefore requires a new physical trace.
        record["independent_benchmark_pair"] = physics_duplicate_of is None
        record["physics_duplicate_of"] = physics_duplicate_of
        record["rgb_duplicate_of"] = rgb_duplicate_of
        if physics_duplicate_of is None:
            seen_physics[physics_signature] = str(resolved)
        if rgb_duplicate_of is None:
            seen_rgb[rgb_signature] = str(resolved)
        formal.append(record)

    independent_count = sum(bool(record["independent_benchmark_pair"]) for record in formal)
    audit = {
        "schema_version": "kinofail.embodiedgen-o4-formal-admission-audit.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": independent_count == len(formal),
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
        "all_formal_pairs_physics_independent": independent_count == len(formal),
        "all_formal_pairs_protocol_bound": all(
            all(record["protocol_binding_checks"].values()) for record in formal
        ),
        "formal_scene_held_out_from_all_calibration_families": all(
            str(record["scene_id"]) not in calibration_scene_ids for record in formal
        ),
        "publication_interpretation": (
            "frozen, held-out, physically independent formal O4 counterfactual evidence"
            if independent_count == len(formal)
            else "duplicate trajectories are deterministic reproducibility evidence only"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
