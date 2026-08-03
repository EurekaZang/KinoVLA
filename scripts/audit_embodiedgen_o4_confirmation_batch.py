#!/usr/bin/env python3
"""Seal a pre-registered multi-scene O4 confirmation batch without dropping rejections."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.sim.embodiedgen_asset import audit_embodiedgen_room_manifest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _hash_check(spec: dict[str, str]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def _scene_attempt(
    request: dict[str, Any],
    *,
    scene_root: Path,
    rejection_root: Path,
    batch_path: Path,
    batch_sha256: str,
) -> dict[str, Any]:
    room = str(request["room_type"])
    scene_seed = int(request["source_scene_seed"])
    formal_seed = int(request["formal_episode_seed"])
    root = scene_root / f"{room}_seed{scene_seed}"
    generation_request_path = scene_root / f"{room}_seed{scene_seed}_request.json"
    source_path = root / "source_manifest.json"
    source_integrity_path = root / "source_integrity_audit.json"
    compiled_path = root / "kinofail/compiled_scene_audit.json"
    rtx_path = root / "kinofail/rtx_qa/rtx_scene_audit.json"
    go2_path = root / "kinofail/go2_qa/go2_scene_audit.json"
    rejection_path = rejection_root / f"{room}_seed{scene_seed}.json"
    formal_manifests = []
    for path in sorted((root / "kinofail/o4_pairs").glob("*/pair_manifest.json")):
        value = _json(path)
        if value.get("protocol_role") == "formal" and int(value.get("seed", -1)) == formal_seed:
            formal_manifests.append((path, value))

    checks: dict[str, bool] = {}
    if generation_request_path.is_file():
        generation = _json(generation_request_path)
        checks.update(
            {
                "generation_request_scene_id": generation.get("scene_id")
                == request["scene_id"],
                "generation_request_room_type": generation.get("room_type") == room,
                "generation_request_seed": int(generation.get("seed", -1)) == scene_seed,
                "generation_request_complexity": generation.get("complexity")
                == request["complexity"],
            }
        )
    else:
        checks["generation_request_present"] = False

    stages: dict[str, Any] = {}
    source_passed = False
    if source_path.is_file():
        source = _json(source_path)
        source_audit = audit_embodiedgen_room_manifest(source_path)
        source_passed = bool(source_audit["passed"])
        checks.update(
            {
                "source_scene_id": source.get("scene_id") == request["scene_id"],
                "source_room_type": source.get("generation", {}).get("room_type") == room,
                "source_seed": int(source.get("generation", {}).get("seed", -1))
                == scene_seed,
                "source_package_integrity": source_passed,
            }
        )
        stages["source"] = {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "passed": source_passed,
            "audit": source_audit,
        }

    compiled = _json(compiled_path) if compiled_path.is_file() else None
    rtx = _json(rtx_path) if rtx_path.is_file() else None
    go2 = _json(go2_path) if go2_path.is_file() else None
    for name, path, value in (
        ("compiled", compiled_path, compiled),
        ("rtx", rtx_path, rtx),
        ("go2", go2_path, go2),
    ):
        if value is not None:
            stages[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "passed": value.get("passed") is True,
            }
    if compiled is not None:
        checks["compiled_scene_id"] = compiled.get("scene_id") == request["scene_id"]
        checks["compiled_binds_source"] = (
            source_path.is_file()
            and compiled.get("source_manifest_sha256") == _sha256(source_path)
        )
    if rtx is not None and compiled is not None:
        checks["rtx_binds_compiled"] = rtx.get("compiled_audit_sha256") == _sha256(
            compiled_path
        )
    if go2 is not None and compiled is not None:
        checks["go2_binds_compiled"] = go2.get("compiled_audit_sha256") == _sha256(
            compiled_path
        )
    if rtx is not None and go2 is not None:
        checks["rtx_go2_same_episode"] = rtx.get("episode_usd_sha256") == go2.get(
            "episode_usd_sha256"
        )

    rejection = _json(rejection_path) if rejection_path.is_file() else None
    if rejection is not None:
        selection = rejection.get("selection_policy", {})
        confirmation_batch = rejection.get("confirmation_batch", {})
        evidence = rejection.get("evidence", {})
        checks.update(
            {
                "rejection_binds_scene": (
            rejection.get("scene_id") == request["scene_id"]
            and int(rejection.get("source_scene_seed", -1)) == scene_seed
            and rejection.get("accepted") is False
                ),
                "rejection_binds_scene_family": rejection.get("scene_family")
                == request["scene_family"],
                "rejection_binds_formal_seed": int(
                    rejection.get("formal_episode_seed", -1)
                )
                == formal_seed,
                "rejection_declares_no_formal_run": rejection.get(
                    "formal_o4_run_executed"
                )
                is False,
                "rejection_counts_in_denominator": selection.get(
                    "counts_against_confirmation_batch"
                )
                is True,
                "rejection_no_replacement_seed": selection.get(
                    "replacement_seed_used"
                )
                is False,
                "rejection_no_threshold_change": selection.get(
                    "qa_threshold_changed"
                )
                is False,
                "rejection_no_code_change": selection.get(
                    "scene_or_operator_code_changed"
                )
                is False,
                "rejection_binds_batch_path": _resolve(
                    confirmation_batch.get("path", "")
                )
                == batch_path,
                "rejection_binds_batch_hash": confirmation_batch.get("sha256")
                == batch_sha256,
                "rejection_reason_nonempty": bool(rejection.get("reason")),
            }
        )
        evidence_paths = {
            "source_manifest": source_path,
            "source_integrity_audit": source_integrity_path,
            "compiled_scene_audit": compiled_path,
            "rtx_scene_audit": rtx_path,
            "go2_scene_audit": go2_path,
        }
        expected_evidence = {
            "source_manifest",
            "source_integrity_audit",
            *( ["compiled_scene_audit"] if compiled is not None else [] ),
            *( ["rtx_scene_audit"] if rtx is not None else [] ),
            *( ["go2_scene_audit"] if go2 is not None else [] ),
        }
        checks["rejection_evidence_stage_set"] = expected_evidence.issubset(evidence)
        for name in sorted(expected_evidence):
            spec = evidence.get(name, {})
            expected_path = evidence_paths[name]
            checks[f"rejection_evidence::{name}::path"] = (
                bool(spec.get("path")) and _resolve(spec["path"]) == expected_path.resolve()
            )
            checks[f"rejection_evidence::{name}::hash"] = (
                expected_path.is_file()
                and spec.get("sha256") == _sha256(expected_path)
            )
        failed_gate = str(rejection.get("failed_gate", ""))
        if "source" in failed_gate:
            checks["rejection_failed_gate_consistent"] = not source_passed
        elif "compiled" in failed_gate or "route" in failed_gate:
            checks["rejection_failed_gate_consistent"] = (
                compiled is not None and compiled.get("passed") is not True
            )
        elif "rtx" in failed_gate:
            checks["rejection_failed_gate_consistent"] = (
                compiled is not None
                and compiled.get("passed") is True
                and rtx is not None
                and rtx.get("passed") is not True
                and go2 is None
            )
        elif "go2" in failed_gate:
            checks["rejection_failed_gate_consistent"] = (
                rtx is not None
                and rtx.get("passed") is True
                and go2 is not None
                and go2.get("passed") is not True
            )
        else:
            checks["rejection_failed_gate_consistent"] = False
        stages["rejection"] = {
            "path": str(rejection_path),
            "sha256": _sha256(rejection_path),
            "failed_gate": rejection.get("failed_gate"),
        }

    if len(formal_manifests) > 1:
        checks["at_most_one_formal_run"] = False
    elif len(formal_manifests) == 1:
        formal_path, formal = formal_manifests[0]
        admission_path = formal_path.parent / "formal_admission_audit.json"
        admission = _json(admission_path) if admission_path.is_file() else None
        checks.update(
            {
                "formal_only_after_scene_qa": go2 is not None and go2.get("passed") is True,
                "formal_scene_id": formal.get("scene_id") == request["scene_id"],
                "formal_seed": int(formal.get("seed", -1)) == formal_seed,
                "formal_pair_passed": formal.get("passed") is True,
                "formal_admission_audit_passed": admission is not None
                and admission.get("passed") is True,
            }
        )
        stages["formal"] = {
            "path": str(formal_path),
            "sha256": _sha256(formal_path),
            "passed": formal.get("passed") is True,
            "admission_audit": None
            if admission is None
            else {"path": str(admission_path), "sha256": _sha256(admission_path)},
        }
    if rejection is not None:
        checks["rejection_has_no_formal_manifest"] = not formal_manifests

    if len(formal_manifests) == 1:
        outcome = "formal_passed" if formal_manifests[0][1].get("passed") is True else "formal_failed"
    elif rejection is not None:
        outcome = "rejected_before_formal"
    elif source_path.is_file() or generation_request_path.is_file():
        outcome = "pending_or_unsealed_failure"
    else:
        outcome = "not_started"
    terminal = outcome in {"formal_passed", "formal_failed", "rejected_before_formal"}
    integrity_passed = all(checks.values()) and terminal
    return {
        "scene_id": request["scene_id"],
        "scene_family": request["scene_family"],
        "source_scene_seed": scene_seed,
        "formal_episode_seed": formal_seed,
        "outcome": outcome,
        "terminal": terminal,
        "integrity_passed": integrity_passed,
        "checks": checks,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=Path,
        default=ROOT / "configs/data/kinofail_embodiedgen_o4_confirmation_batch_v1.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument(
        "--parent-formal-manifest",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/Kitchen_seed20260727/kinofail/o4_pairs/formal_heldout_seed20260728_v4/pair_manifest.json",
    )
    parser.add_argument(
        "--parent-admission-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/Kitchen_seed20260727/kinofail/o4_pairs/formal_heldout_seed20260728_v4/formal_admission_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_confirmation_batch_v1_audit.json",
    )
    args = parser.parse_args()

    batch_path = args.batch.resolve()
    batch = _json(batch_path)
    batch_sha256 = _sha256(batch_path)
    frozen_checks = {
        f"code::{name}": _hash_check(spec) for name, spec in batch["frozen_code"].items()
    }
    frozen_checks.update(
        {
            f"calibration::{row['scene_id']}": _resolve(row["path"]).is_file()
            and _sha256(_resolve(row["path"])) == row["sha256"]
            for row in batch["calibration_exclusions"]
        }
    )
    parent_manifest = _json(args.parent_formal_manifest.resolve())
    parent_admission = _json(args.parent_admission_audit.resolve())
    parent_checks = {
        "formal_pair_passed": parent_manifest.get("passed") is True,
        "admission_audit_passed": parent_admission.get("passed") is True,
        "scene_id": parent_manifest.get("scene_id")
        == batch["parent_formal_result"]["scene_id"],
        "formal_seed": int(parent_manifest.get("seed", -1))
        == int(batch["parent_formal_result"]["formal_episode_seed"]),
        "protocol_hash": _hash_check(batch["parent_formal_result"]["protocol"]),
    }
    attempts = [
        _scene_attempt(
            request,
            scene_root=args.scene_root.resolve(),
            rejection_root=args.scene_root.resolve() / "rejections",
            batch_path=batch_path,
            batch_sha256=batch_sha256,
        )
        for request in batch["new_scene_requests"]
    ]
    sealed = all(bool(attempt["terminal"]) for attempt in attempts)
    audit_integrity_passed = (
        all(frozen_checks.values())
        and all(parent_checks.values())
        and sealed
        and all(bool(attempt["integrity_passed"]) for attempt in attempts)
    )
    new_passes = sum(attempt["outcome"] == "formal_passed" for attempt in attempts)
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-confirmation-batch-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": batch_sha256},
        "sealed": sealed,
        "audit_integrity_passed": audit_integrity_passed,
        "all_requested_new_scenes_formally_passed": new_passes == len(attempts),
        "three_heldout_scene_family_target_met": 1 + new_passes >= 3,
        "heldout_scene_families": {
            "requested_total_including_parent": 1 + len(attempts),
            "formally_passed_total_including_parent": 1 + new_passes,
            "rejected_or_failed_new_scenes": sum(
                attempt["outcome"] in {"formal_failed", "rejected_before_formal"}
                for attempt in attempts
            ),
            "pending_new_scenes": sum(not bool(attempt["terminal"]) for attempt in attempts),
        },
        "frozen_input_checks": frozen_checks,
        "parent_formal_result_checks": parent_checks,
        "attempts": attempts,
        "interpretation": (
            "Scene generation/QA rejection is counted in the denominator. Only a passed pair "
            "with an independent formal-admission audit adds a held-out scene family."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not audit_integrity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
