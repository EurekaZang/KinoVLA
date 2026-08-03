#!/usr/bin/env python3
"""Audit realistic-route locomotion training artifacts and final learning scalars."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_scalar(events: EventAccumulator, tag: str) -> float:
    values = events.Scalars(tag)
    if not values:
        raise ValueError(f"missing scalar: {tag}")
    return float(values[-1].value)


def main() -> int:
    parser = argparse.ArgumentParser()
    out_dir = ROOT / "outputs/locomotion/realistic_route_v1"
    parser.add_argument("--manifest", type=Path, default=out_dir / "training_manifest.json")
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "configs/locomotion/go2_realistic_route_ppo_v1_freeze.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT / "outputs/eval/realistic_route_locomotion_v1_training_preflight.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/eval/realistic_route_locomotion_v1_training_postrun.json",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    freeze_path = args.freeze.resolve()
    preflight_path = args.preflight.resolve()
    manifest = _json(manifest_path)
    freeze = _json(freeze_path)
    preflight = _json(preflight_path)
    event_files = sorted(manifest_path.parent.glob("**/events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected one event file, got {event_files}")
    events = EventAccumulator(str(event_files[0]))
    events.Reload()
    artifacts = manifest.get("artifacts", {})
    artifact_checks: dict[str, bool] = {}
    for name, spec in artifacts.items():
        path = Path(spec["path"])
        artifact_checks[f"artifact_{name}_hash"] = path.is_file() and _sha256(path) == spec[
            "sha256"
        ]
    stable = Path(artifacts["stable_policy"]["path"])
    exported = Path(artifacts["exported_policy"]["path"])
    legacy = freeze["preserved_legacy_artifacts"]
    checks = {
        "training_preflight_passed": preflight.get("passed") is True,
        "training_manifest_passed": manifest.get("passed") is True,
        "frozen_scale_completed": manifest.get("num_envs") == 4096
        and manifest.get("max_iterations") == 1200
        and manifest.get("seed") == 20261130,
        "stable_policy_matches_export": _sha256(stable) == _sha256(exported),
        "legacy_shipped_policy_preserved": _sha256(
            ROOT / legacy["shipped_policy"]["path"]
        )
        == legacy["shipped_policy"]["sha256"],
        "legacy_direct_policy_preserved": _sha256(
            ROOT / legacy["direct_velocity_policy"]["path"]
        )
        == legacy["direct_velocity_policy"]["sha256"],
        "training_not_a0_a7_evidence": manifest.get("evidence_boundary", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False,
        "post_training_gates_still_required": manifest.get("evidence_boundary", {}).get(
            "requires_post_training_route_gates"
        )
        is True,
        **artifact_checks,
    }
    final_scalars = {
        "mean_reward": _last_scalar(events, "Train/mean_reward"),
        "mean_episode_length_steps": _last_scalar(events, "Train/mean_episode_length"),
        "linear_velocity_error_mps": _last_scalar(
            events, "Metrics/base_velocity/error_vel_xy"
        ),
        "yaw_velocity_error_radps": _last_scalar(
            events, "Metrics/base_velocity/error_vel_yaw"
        ),
        "time_out_fraction": _last_scalar(events, "Episode_Termination/time_out"),
        "base_contact_fraction": _last_scalar(
            events, "Episode_Termination/base_contact"
        ),
        "mean_action_noise_std": _last_scalar(events, "Policy/mean_noise_std"),
    }
    result = {
        "schema_version": "kinofail.realistic-route-locomotion-training-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "policy_admitted_for_collection": False,
        "checks": checks,
        "final_training_scalars": final_scalars,
        "scientific_boundary": (
            "Training completion and healthy aggregate scalars do not establish low-speed "
            "route performance; the frozen policy remains inadmissible until independent "
            "multi-seed inference and realistic-scene route gates pass."
        ),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "freeze": {"path": str(freeze_path), "sha256": _sha256(freeze_path)},
            "preflight": {"path": str(preflight_path), "sha256": _sha256(preflight_path)},
            "tensorboard_events": {
                "path": str(event_files[0]),
                "sha256": _sha256(event_files[0]),
            },
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(output),
                "audit_passed": result["audit_passed"],
                "policy_admitted_for_collection": False,
                "final_training_scalars": final_scalars,
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
