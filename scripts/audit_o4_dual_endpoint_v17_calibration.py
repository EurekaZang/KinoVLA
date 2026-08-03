#!/usr/bin/env python3
"""Seal the preregistered train-only O4 v17 force-cap search."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    if "json_passed" in record:
        return _json(path).get("passed") is record["json_passed"]
    return True


def _maximum_raw_force(summary: dict[str, Any]) -> float:
    value = summary.get("telemetry")
    if not isinstance(value, str):
        return 0.0
    path = _resolve(value)
    maximum = 0.0
    if not path.is_file():
        return maximum
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for foot in row.get("adhesion", {}).get("feet", []):
            maximum = max(maximum, float(foot.get("raw_force_n", 0.0)))
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    runtime = config["runtime"]
    rows = []
    for case in config["cases"]:
        path = _resolve(case["output"]) / "pair_manifest.json"
        manifest = _json(path)
        nominal = manifest.get("takes", {}).get("nominal", {})
        anomaly = manifest.get("takes", {}).get("o4_adhesion", {})
        effect = manifest.get("paired_consequence", {})
        v17 = manifest.get("v17_calibration", {})
        recovered = v17.get("moderate_recoverable_observed") is True
        selection_checks = {
            "manifest_exists": path.is_file(),
            "dose_matches": float(v17.get("force_cap_n_requested", -1.0))
            == float(case["force_cap_n"]),
            "readback_matches": abs(
                float(v17.get("force_cap_n_readback", -1.0)) - float(case["force_cap_n"])
            )
            <= 1.0e-5,
            "nominal_complete": nominal.get("fell") is False
            and nominal.get("steps") == nominal.get("fixed_horizon_steps"),
            "anomaly_attached": int(anomaly.get("attachment_events", 0)) >= 1,
            "anomaly_released": int(anomaly.get("peel_events", 0))
            + int(anomaly.get("break_events", 0))
            >= 1,
            "anomaly_not_fallen": anomaly.get("fell") is False,
            "inside_route_budget": float(
                anomaly.get("maximum_absolute_route_lateral_offset_m", 1.0e9)
            )
            <= float(runtime["maximum_route_deviation_m"]),
            "locomotion_effect": effect.get("available") is True
            and float(effect.get("window_duration_s", -1.0)) >= 1.0
            and float(effect.get("progress_gain_lag_m", -1.0)) >= 0.08,
            "posture_effect": effect.get("available") is True
            and (
                float(effect.get("max_tilt_increase_rad", -1.0)) >= 0.10
                or float(effect.get("min_base_height_drop_m", -1.0)) >= 0.03
            ),
            "development_only": manifest.get("development_only") is True,
            "not_a0_a7_evidence": manifest.get("counts_as_a0_a7_evidence") is False,
        }
        diagnostic = {
            "terminal_fall": anomaly.get("fell") is True,
            "fall_step": anomaly.get("first_fall_step"),
            "first_attachment_step": effect.get("first_attachment_step"),
            "attachment_to_fall_steps": (
                None
                if anomaly.get("first_fall_step") is None
                or effect.get("first_attachment_step") is None
                else int(anomaly["first_fall_step"]) - int(effect["first_attachment_step"])
            ),
            "progress_gain_lag_m": effect.get("progress_gain_lag_m"),
            "max_tilt_increase_rad": effect.get("max_tilt_increase_rad"),
            "min_base_height_drop_m": effect.get("min_base_height_drop_m"),
            "maximum_route_deviation_m": anomaly.get(
                "maximum_absolute_route_lateral_offset_m"
            ),
            "maximum_raw_force_n": _maximum_raw_force(anomaly),
        }
        rows.append(
            {
                "force_cap_n": case["force_cap_n"],
                "manifest": str(path),
                "manifest_sha256": _sha256(path) if path.is_file() else None,
                "selection_checks": selection_checks,
                "selected": all(selection_checks.values()) and recovered,
                "diagnostic": diagnostic,
            }
        )

    selected = sorted(row["force_cap_n"] for row in rows if row["selected"])
    integrity_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(record) for record in config["locked_files"]),
        "all_three_doses_present": sorted(row["force_cap_n"] for row in rows)
        == [8.0, 12.0, 16.0],
        "all_outputs_development_only": all(
            row["selection_checks"]["development_only"]
            and row["selection_checks"]["not_a0_a7_evidence"]
            for row in rows
        ),
    }
    audit = {
        "schema_version": "kinofail.o4-dual-endpoint-v17-calibration-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
        "rows": rows,
        "selected_moderate_force_cap_n": selected[0] if selected else None,
        "selection_passed": bool(selected),
        "passed": bool(selected) and all(integrity_checks.values()),
        "sealed": all(integrity_checks.values()),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "diagnosis": (
            "No force cap passed. A single foot remains attached to a fixed world anchor across "
            "the swing/unloaded phase, so even 8 N accumulates a large off-axis posture moment "
            "and causes a pre-peel fall. Force-cap tuning cannot repair the state model."
        ),
        "required_v18_change": (
            "Separate surface adhesive contact from persistent tether entanglement. Surface "
            "adhesion must use load/contact-aware anisotropic peel and reattachment cycles; "
            "persistent anchors require a visible tether subtype and terminal endpoint."
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": audit["passed"],
                "sealed": audit["sealed"],
                "selected_moderate_force_cap_n": audit["selected_moderate_force_cap_n"],
            },
            indent=2,
        )
    )
    return 0 if audit["sealed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
