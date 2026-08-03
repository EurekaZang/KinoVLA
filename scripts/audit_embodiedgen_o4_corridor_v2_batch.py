#!/usr/bin/env python3
"""Audit a pre-registered robust-corridor O4 scene batch, including all rejections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _hash_spec(spec: dict[str, str]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def _attempt(
    request: dict[str, Any],
    *,
    batch: dict[str, Any],
    scene_root: Path,
) -> dict[str, Any]:
    room = str(request["room_type"])
    source_seed = int(request["source_scene_seed"])
    formal_seed = int(request["formal_episode_seed"])
    root = scene_root / f"{room}_seed{source_seed}"
    generation_path = scene_root / f"{room}_seed{source_seed}_request.json"
    source_path = root / "source_manifest.json"
    source_integrity_path = root / "source_integrity_audit.json"
    base_path = root / "kinofail_base_v1/compiled_scene_audit.json"
    corridor_root = root / "kinofail_corridor_v2_tracking030"
    compiled_path = corridor_root / "compiled_scene_audit.json"
    corridor_rejection_path = corridor_root / "corridor_refinement_rejection.json"
    rtx_path = corridor_root / "rtx_qa/rtx_scene_audit.json"
    go2_path = corridor_root / "go2_qa/go2_scene_audit.json"
    admission_path = corridor_root / "corridor_v2_admission_audit.json"
    manual_rejection_path = scene_root / "rejections_v2" / f"{room}_seed{source_seed}.json"
    checks: dict[str, bool] = {}
    stages: dict[str, Any] = {}

    generation = _json(generation_path) if generation_path.is_file() else None
    if generation is not None:
        checks.update(
            {
                "generation_scene_id": generation.get("scene_id") == request["scene_id"],
                "generation_room_type": generation.get("room_type") == room,
                "generation_seed": int(generation.get("seed", -1)) == source_seed,
                "generation_complexity": generation.get("complexity")
                == request["complexity"],
            }
        )
        stages["generation_request"] = {
            "path": str(generation_path),
            "sha256": _sha256(generation_path),
        }

    source = _json(source_path) if source_path.is_file() else None
    source_audit = None
    if source is not None:
        source_audit = audit_embodiedgen_room_manifest(source_path)
        checks.update(
            {
                "source_scene_id": source.get("scene_id") == request["scene_id"],
                "source_room_type": source.get("generation", {}).get("room_type") == room,
                "source_seed": int(source.get("generation", {}).get("seed", -1))
                == source_seed,
                "source_package_integrity": source_audit["passed"] is True,
                "source_integrity_record_present": source_integrity_path.is_file(),
                "source_integrity_record_matches_recomputation": source_integrity_path.is_file()
                and _json(source_integrity_path) == source_audit,
            }
        )
        stages["source"] = {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "passed": source_audit["passed"],
            "file_count": source_audit["file_count"],
            "texture_file_count": source_audit["texture_file_count"],
            "total_bytes": source_audit["total_bytes"],
        }

    base = _json(base_path) if base_path.is_file() else None
    if base is not None:
        checks.update(
            {
                "base_scene_id": base.get("scene_id") == request["scene_id"],
                "base_binds_source": source_path.is_file()
                and base.get("source_manifest_sha256") == _sha256(source_path),
            }
        )
        stages["base_compile"] = {
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "passed": base.get("passed") is True,
        }

    compiled = _json(compiled_path) if compiled_path.is_file() else None
    corridor_rejection = (
        _json(corridor_rejection_path) if corridor_rejection_path.is_file() else None
    )
    if corridor_rejection is not None:
        checks.update(
            {
                "corridor_rejection_binds_base": base_path.is_file()
                and corridor_rejection.get("base_compiled_audit", {}).get("sha256")
                == _sha256(base_path),
                "corridor_rejection_no_formal_authority": corridor_rejection.get(
                    "formal_operator_run_authorized"
                )
                is False,
                "corridor_rejection_reason_nonempty": bool(
                    corridor_rejection.get("reason")
                ),
            }
        )
        stages["corridor_refinement"] = {
            "path": str(corridor_rejection_path),
            "sha256": _sha256(corridor_rejection_path),
            "passed": False,
        }
    if compiled is not None:
        route = compiled.get("route", {})
        contract = batch["corridor_contract"]
        episode_path = corridor_root / "episode_v2.usda"
        checks.update(
            {
                "corridor_schema_v2": compiled.get("schema_version")
                == "kinofail.embodiedgen-compiled-scene.v2",
                "corridor_passed": compiled.get("passed") is True,
                "corridor_scene_id": compiled.get("scene_id") == request["scene_id"],
                "corridor_binds_base": base_path.is_file()
                and compiled.get("base_compiled_audit_sha256") == _sha256(base_path),
                "corridor_binds_source": source_path.is_file()
                and compiled.get("source_manifest_sha256") == _sha256(source_path),
                "corridor_tracking_budget": math.isclose(
                    float(route.get("maximum_admissible_tracking_error_m", math.nan)),
                    float(contract["maximum_tracking_error_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ),
                "corridor_required_clearance": math.isclose(
                    float(route.get("robust_centerline_clearance_m", math.nan)),
                    float(contract["required_centerline_clearance_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ),
                "corridor_measured_clearance": float(
                    route.get("measured_min_clearance_m", -math.inf)
                )
                + 1.0e-6
                >= float(contract["required_centerline_clearance_m"]),
                "corridor_length": math.isclose(
                    float(route.get("route_length_m", math.nan)),
                    float(contract["corridor_length_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ),
                "corridor_episode_hash": episode_path.is_file()
                and compiled.get("files", {}).get("episode_v2.usda")
                == _sha256(episode_path),
            }
        )
        stages["corridor_refinement"] = {
            "path": str(compiled_path),
            "sha256": _sha256(compiled_path),
            "passed": compiled.get("passed") is True,
        }

    rtx = _json(rtx_path) if rtx_path.is_file() else None
    go2 = _json(go2_path) if go2_path.is_file() else None
    admission = _json(admission_path) if admission_path.is_file() else None
    episode_path = corridor_root / "episode_v2.usda"
    for name, path, value in (
        ("rtx", rtx_path, rtx),
        ("go2", go2_path, go2),
        ("corridor_admission", admission_path, admission),
    ):
        if value is not None:
            stages[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "passed": value.get("passed") is True,
            }
    if rtx is not None:
        checks["rtx_binds_corridor"] = compiled_path.is_file() and rtx.get(
            "compiled_audit_sha256"
        ) == _sha256(compiled_path)
        checks["rtx_binds_episode"] = episode_path.is_file() and rtx.get(
            "episode_usd_sha256"
        ) == _sha256(episode_path)
    if go2 is not None:
        checks["go2_binds_corridor"] = compiled_path.is_file() and go2.get(
            "compiled_audit_sha256"
        ) == _sha256(compiled_path)
        checks["go2_binds_episode"] = episode_path.is_file() and go2.get(
            "episode_usd_sha256"
        ) == _sha256(episode_path)
    if admission is not None:
        checks["admission_binds_corridor"] = admission.get("compiled_audit", {}).get(
            "sha256"
        ) == _sha256(compiled_path)
        checks["admission_binds_rtx"] = rtx_path.is_file() and admission.get(
            "rtx_audit", {}
        ).get("sha256") == _sha256(rtx_path)
        checks["admission_binds_go2"] = go2_path.is_file() and admission.get(
            "go2_audit", {}
        ).get("sha256") == _sha256(go2_path)

    formal_manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((corridor_root / "o4_pairs").glob("*/pair_manifest.json")):
        value = _json(path)
        if value.get("protocol_role") == "formal" and int(value.get("seed", -1)) == formal_seed:
            formal_manifests.append((path, value))
    checks["at_most_one_formal_run"] = len(formal_manifests) <= 1
    if len(formal_manifests) == 1:
        formal_path, formal = formal_manifests[0]
        independence_path = formal_path.parent / "formal_admission_audit.json"
        independence = _json(independence_path) if independence_path.is_file() else None
        formal_protocol = formal.get("formal_protocol", {})
        protocol_path = Path(formal_protocol.get("path", "")).resolve()
        run = batch["run_contract"]
        checks.update(
            {
                "formal_only_after_corridor_admission": admission is not None
                and admission.get("passed") is True,
                "formal_scene_id": formal.get("scene_id") == request["scene_id"],
                "formal_seed": int(formal.get("seed", -1)) == formal_seed,
                "formal_compiled_hash": formal.get("compiled_audit_sha256")
                == _sha256(compiled_path),
                "formal_episode_hash": formal.get("episode_usd_sha256")
                == _sha256(episode_path),
                "formal_protocol_present": protocol_path.is_file(),
                "formal_protocol_hash": protocol_path.is_file()
                and formal_protocol.get("sha256") == _sha256(protocol_path),
                "formal_forward_s": math.isclose(
                    float(formal.get("protocol", {}).get("forward_s", math.nan)),
                    float(run["forward_s"]),
                ),
                "formal_peel_s": math.isclose(
                    float(formal.get("protocol", {}).get("peel_pulse_s", math.nan)),
                    float(run["peel_pulse_s"]),
                ),
                "formal_recovery_s": math.isclose(
                    float(
                        formal.get("protocol", {}).get(
                            "post_peel_recovery_s", math.nan
                        )
                    ),
                    float(run["recovery_s"]),
                ),
                "formal_reverse_s": math.isclose(
                    float(formal.get("protocol", {}).get("reverse_retreat_s", math.nan)),
                    float(run["reverse_s"]),
                ),
                "formal_stop_s": math.isclose(
                    float(formal.get("protocol", {}).get("stop_s", math.nan)),
                    float(run["stop_s"]),
                ),
                "formal_result_is_boolean": isinstance(formal.get("passed"), bool),
                "formal_adjudication_audit_consistent": (
                    formal.get("passed") is False
                    or (
                        independence is not None
                        and independence.get("passed") is True
                    )
                ),
            }
        )
        stages["formal"] = {
            "path": str(formal_path),
            "sha256": _sha256(formal_path),
            "passed": formal.get("passed") is True,
            "independence_audit": None
            if independence is None
            else {"path": str(independence_path), "sha256": _sha256(independence_path)},
        }

    manual_rejection = (
        _json(manual_rejection_path) if manual_rejection_path.is_file() else None
    )
    if manual_rejection is not None:
        checks["manual_rejection_binds_scene"] = (
            manual_rejection.get("scene_id") == request["scene_id"]
            and int(manual_rejection.get("source_scene_seed", -1)) == source_seed
            and manual_rejection.get("accepted") is False
        )
        checks["manual_rejection_no_formal"] = not formal_manifests
        stages["manual_rejection"] = {
            "path": str(manual_rejection_path),
            "sha256": _sha256(manual_rejection_path),
        }

    if formal_manifests:
        outcome = (
            "formal_passed"
            if formal_manifests[0][1].get("passed") is True
            else "formal_failed"
        )
    elif manual_rejection is not None:
        outcome = "manual_rejection"
    elif source_audit is not None and source_audit["passed"] is not True:
        outcome = "source_rejected"
    elif base is not None and base.get("passed") is not True:
        outcome = "base_compile_rejected"
    elif corridor_rejection is not None:
        outcome = "corridor_refinement_rejected"
        checks["no_downstream_after_corridor_rejection"] = (
            rtx is None and go2 is None and admission is None and not formal_manifests
        )
    elif rtx is not None and rtx.get("passed") is not True:
        outcome = "rtx_rejected"
        checks["no_downstream_after_rtx_rejection"] = (
            go2 is None and admission is None and not formal_manifests
        )
    elif go2 is not None and go2.get("passed") is not True:
        outcome = "go2_rejected"
        checks["no_downstream_after_go2_rejection"] = (
            admission is None and not formal_manifests
        )
    elif admission is not None and admission.get("passed") is not True:
        outcome = "corridor_admission_rejected"
        checks["no_formal_after_admission_rejection"] = not formal_manifests
    else:
        outcome = "pending"
    terminal = outcome != "pending"
    return {
        "scene_id": request["scene_id"],
        "scene_instance_family": request["scene_instance_family"],
        "room_type": room,
        "source_scene_seed": source_seed,
        "formal_episode_seed": formal_seed,
        "outcome": outcome,
        "terminal": terminal,
        "integrity_passed": terminal and bool(checks) and all(checks.values()),
        "checks": checks,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_o4_corridor_v2_extension_batch_v1.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_corridor_v2_extension_batch_v1_audit.json",
    )
    args = parser.parse_args()
    batch_path = args.batch.resolve()
    batch = _json(batch_path)
    frozen_checks = {
        f"code::{name}": _hash_spec(spec) for name, spec in batch["frozen_code"].items()
    }
    development_checks = {
        f"development::{name}": _hash_spec(spec)
        for name, spec in batch["development_evidence"].items()
    }
    exclusion_checks = {
        f"exclusion::{row['scene_id']}": _hash_spec(row)
        for row in batch["calibration_exclusions"]
    }
    attempts = [
        _attempt(request, batch=batch, scene_root=args.scene_root.resolve())
        for request in batch["new_scene_requests"]
    ]
    sealed = all(attempt["terminal"] for attempt in attempts)
    integrity = (
        all(frozen_checks.values())
        and all(development_checks.values())
        and all(exclusion_checks.values())
        and sealed
        and all(attempt["integrity_passed"] for attempt in attempts)
    )
    formal_passes = sum(attempt["outcome"] == "formal_passed" for attempt in attempts)
    payload = {
        "schema_version": "kinofail.embodiedgen-o4-corridor-v2-batch-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "requested_scene_instances": len(attempts),
        "formal_passes": formal_passes,
        "primary_target_met": formal_passes
        >= int(batch["inference_scope"]["minimum_formal_passes_for_primary_target"]),
        "frozen_input_checks": frozen_checks,
        "development_evidence_checks": development_checks,
        "calibration_exclusion_checks": exclusion_checks,
        "attempts": attempts,
        "interpretation": (
            "Every pre-registered scene remains in the denominator. Only a passed formal pair "
            "with a passed independent audit contributes a confirmatory O4 scene instance."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
