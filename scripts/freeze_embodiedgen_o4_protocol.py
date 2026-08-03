#!/usr/bin/env python3
"""Freeze the held-out formal protocol for one admitted EmbodiedGen O4 scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_file(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _coordinate_free_operator_contract(calibration: dict[str, object]) -> dict[str, object]:
    """Return O4 physics parameters while excluding scene-frame coordinates.

    The patch polygon, floor elevation, and route axis must vary across generated rooms.  Their
    *construction rule* is checked separately through each compiled-scene placement contract.
    """

    operator = calibration["operator"]
    parameters = {
        key: value
        for key, value in operator["parameters"].items()
        if key not in ("surface_z_m", "progress_axis_xy")
    }
    return {
        "id": operator["id"],
        "name": operator["name"],
        "fidelity": operator["fidelity"],
        "parameters": parameters,
        "scene_coordinate_fields": {
            "region_xy_m": "derived from compiled route-relative placement",
            "surface_z_m": "derived from compiled scene floor",
            "progress_axis_xy": "derived from local route direction",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="At least two passed calibration manifests from distinct scene families/seeds.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--formal-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--forward-s", type=float, required=True)
    parser.add_argument("--peel-pulse-s", type=float, default=0.10)
    parser.add_argument("--recovery-s", type=float, default=1.00)
    parser.add_argument("--reverse-s", type=float, default=3.0)
    parser.add_argument("--stop-s", type=float, default=1.0)
    parser.add_argument("--rgb-fps", type=float, default=5.0)
    args = parser.parse_args()

    episode = args.episode_usd.resolve()
    compiled_path = args.compiled_audit.resolve()
    calibration_paths = [path.resolve() for path in args.calibration_manifest]
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    calibrations = [json.loads(path.read_text(encoding="utf-8")) for path in calibration_paths]
    source_manifest = json.loads(Path(compiled["source_manifest"]).read_text(encoding="utf-8"))
    room_type = str(source_manifest["generation"]["room_type"])
    source_scene_seed = int(source_manifest["generation"]["seed"])
    if not compiled.get("passed"):
        raise RuntimeError("cannot freeze a scene whose compiled audit failed")
    if len(calibrations) < 2:
        raise RuntimeError("formal freeze requires at least two calibration scenes")
    for calibration in calibrations:
        if not calibration.get("passed") or calibration.get("protocol_role") != "calibration":
            raise RuntimeError(
                "cannot freeze without passed, explicitly marked calibration pairs"
            )
        if calibration.get("schema_version") != "kinofail.embodiedgen-o4-paired-sequence.v4":
            raise RuntimeError(
                "calibration pair does not use the current registered-terminal-outcome protocol"
            )
    calibration_scene_ids = {str(value["scene_id"]) for value in calibrations}
    calibration_scene_seeds = {int(value["source_scene_seed"]) for value in calibrations}
    calibration_families = {str(value["scene_family"]) for value in calibrations}
    if len(calibration_scene_ids) != len(calibrations):
        raise RuntimeError("calibration manifests must come from distinct scenes")
    if len(calibration_scene_seeds) != len(calibrations):
        raise RuntimeError("calibration manifests must use distinct generated-scene seeds")
    if len(calibration_families) < 2:
        raise RuntimeError("calibration set must cover at least two distinct scene families")
    required_placement_contract = "route_relative_from_collector_start_v2"
    calibration_compiled_audits = [
        json.loads(Path(value["compiled_audit"]).read_text(encoding="utf-8"))
        for value in calibrations
    ]
    placement_contracts = [
        value["operator"].get("placement_contract") for value in calibration_compiled_audits
    ]
    if compiled["operator"].get("placement_contract") != required_placement_contract:
        raise RuntimeError("formal scene does not use the current route-relative placement")
    if any(value != required_placement_contract for value in placement_contracts):
        raise RuntimeError("at least one calibration uses a stale O4 placement contract")
    if args.scene_seed != source_scene_seed:
        raise ValueError(
            f"declared scene seed {args.scene_seed} differs from source seed {source_scene_seed}"
        )
    if (
        compiled.get("scene_id") in calibration_scene_ids
        or source_scene_seed in calibration_scene_seeds
    ):
        raise RuntimeError("formal scene must be held out from protocol calibration")
    if len(set(args.formal_seeds)) != len(args.formal_seeds):
        raise ValueError("formal seeds must be unique")
    calibration_episode_seeds = {int(value["seed"]) for value in calibrations}
    if calibration_episode_seeds & set(args.formal_seeds):
        raise ValueError("calibration episode seeds cannot be reused for formal collection")

    rtx_audit = episode.parent / "rtx_qa/rtx_scene_audit.json"
    go2_audit = episode.parent / "go2_qa/go2_scene_audit.json"
    episode_hash = _sha256(episode)
    for name, audit_path in (("RTX", rtx_audit), ("Go2", go2_audit)):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("passed") or audit.get("episode_usd_sha256") != episode_hash:
            raise RuntimeError(f"{name} audit is failed or stale: {audit_path}")
    existing_o4_files = list((episode.parent / "o4_pairs").glob("*/*"))
    if existing_o4_files:
        raise RuntimeError(
            "formal scene is no longer held out; existing O4 evidence: "
            + ", ".join(str(path) for path in existing_o4_files[:8])
        )
    requested_run_contract = {
        "forward_s": float(args.forward_s),
        "peel_pulse_s": float(args.peel_pulse_s),
        "recovery_s": float(args.recovery_s),
        "reverse_s": float(args.reverse_s),
        "stop_s": float(args.stop_s),
        "rgb_fps": float(args.rgb_fps),
    }
    calibrated_run_contracts = [
        {
            "forward_s": float(calibration["protocol"]["forward_s"]),
            "peel_pulse_s": float(calibration["protocol"]["peel_pulse_s"]),
            "recovery_s": float(calibration["protocol"]["post_peel_recovery_s"]),
            "reverse_s": float(calibration["protocol"]["reverse_retreat_s"]),
            "stop_s": float(calibration["protocol"]["stop_s"]),
            "rgb_fps": float(calibration["protocol"]["rgb_fps"]),
        }
        for calibration in calibrations
    ]
    if any(contract != requested_run_contract for contract in calibrated_run_contracts):
        raise RuntimeError(
            "formal run contract differs from at least one passed calibration: "
            f"requested={requested_run_contract}, calibrated={calibrated_run_contracts}"
        )
    operator_contracts = [_coordinate_free_operator_contract(value) for value in calibrations]
    if any(contract != operator_contracts[0] for contract in operator_contracts[1:]):
        raise RuntimeError("O4 operator contract differs across calibration scenes")
    consequence_contracts = [
        calibration["paired_consequence"]["thresholds"] for calibration in calibrations
    ]
    if any(contract != consequence_contracts[0] for contract in consequence_contracts[1:]):
        raise RuntimeError("paired-consequence thresholds differ across calibration scenes")
    terminal_contracts = [calibration["terminal_outcome"]["contract"] for calibration in calibrations]
    if any(contract != terminal_contracts[0] for contract in terminal_contracts[1:]):
        raise RuntimeError("terminal-outcome contract differs across calibration scenes")
    live_collector_hash = _sha256(ROOT / "scripts/isaac_collect_embodiedgen_o4_pair.py")
    live_backend_hash = _sha256(ROOT / "kino_vla/sim/isaac_policy_backend.py")
    live_adhesion_hash = _sha256(ROOT / "kino_vla/sim/adhesion.py")
    for calibration in calibrations:
        if calibration["provenance"].get("collector_sha256") != live_collector_hash:
            raise RuntimeError("collector changed after at least one passed calibration")
        if calibration["provenance"].get("backend_sha256") != live_backend_hash:
            raise RuntimeError("backend changed after at least one passed calibration")
        if calibration["provenance"].get("adhesion_model_sha256") != live_adhesion_hash:
            raise RuntimeError("adhesion model changed after at least one passed calibration")
    protocol = {
        "schema_version": "kinofail.embodiedgen-o4-formal-protocol.v3",
        "protocol_id": f"embodiedgen-{room_type.lower()}-{args.scene_seed}-o4-v4",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "scene_id": compiled["scene_id"],
        "scene_family": f"EmbodiedGen-v2-{room_type}",
        "scene_seed": int(args.scene_seed),
        "formal_episode_seeds": [int(seed) for seed in args.formal_seeds],
        "episode_seed_semantics": (
            "stochastic reset/physics replicate inside one fixed generated scene; "
            "these seeds are not independent scene-family coverage"
        ),
        "run_contract": requested_run_contract,
        "operator_contract": operator_contracts[0],
        "consequence_contract": consequence_contracts[0],
        "terminal_outcome_contract": terminal_contracts[0],
        "calibration_contract": {
            "minimum_distinct_scenes": 2,
            "minimum_distinct_scene_families": 2,
            "observed_distinct_scenes": len(calibration_scene_ids),
            "observed_distinct_scene_families": len(calibration_families),
            "scene_families": sorted(calibration_families),
            "shared_run_contract_verified": True,
            "shared_operator_contract_verified": True,
            "shared_consequence_contract_verified": True,
            "shared_terminal_outcome_contract_verified": True,
            "shared_route_relative_placement_contract_verified": True,
            "placement_contract": required_placement_contract,
        },
        "calibration_exclusions": [
            {
                "seed": int(calibration["seed"]),
                "scene_id": calibration["scene_id"],
                "scene_family": calibration["scene_family"],
                "source_scene_seed": int(calibration["source_scene_seed"]),
                "manifest": _frozen_file(path),
                "reason": (
                    "used to calibrate O4 parameters/outcome contracts; never formal evidence"
                ),
            }
            for calibration, path in zip(calibrations, calibration_paths, strict=True)
        ],
        "heldout_scene_contract": {
            "no_prior_o4_pair_manifests_at_freeze": True,
            "scene_differs_from_calibration": True,
            "source_scene_seed_verified": True,
        },
        "frozen_files": {
            "collector": _frozen_file(ROOT / "scripts/isaac_collect_embodiedgen_o4_pair.py"),
            "backend": _frozen_file(ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "adhesion_model": _frozen_file(ROOT / "kino_vla/sim/adhesion.py"),
            "episode_usd": _frozen_file(episode),
            "compiled_audit": _frozen_file(compiled_path),
            "rtx_audit": _frozen_file(rtx_audit),
            "go2_audit": _frozen_file(go2_audit),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {args.out}")
    args.out.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out.resolve())
    print(_sha256(args.out.resolve()))


if __name__ == "__main__":
    main()
