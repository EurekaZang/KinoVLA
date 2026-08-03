#!/usr/bin/env python3
"""Seal the frozen two-point O4 surface-adhesion v19 strength calibration."""

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
    return "json_passed" not in record or _json(path).get("passed") is record["json_passed"]


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
    rows = []
    for case in config["cases"]:
        path = _resolve(case["output"]) / "pair_manifest.json"
        manifest = _json(path)
        surface = manifest.get("v18_surface_calibration", {})
        checks = surface.get("checks", {})
        anomaly = manifest.get("takes", {}).get("o4_adhesion", {})
        effect = manifest.get("paired_consequence", {})
        last = surface.get("last_telemetry", {})
        rows.append(
            {
                "tangential_force_cap_n": case["tangential_force_cap_n"],
                "manifest": str(path),
                "manifest_sha256": _sha256(path) if path.is_file() else None,
                "reported_passed": manifest.get("passed"),
                "checks": checks,
                "selected": manifest.get("passed") is True,
                "measurements": {
                    "fell": anomaly.get("fell"),
                    "first_fall_step": anomaly.get("first_fall_step"),
                    "attachment_events": anomaly.get("attachment_events"),
                    "peel_events": anomaly.get("peel_events"),
                    "peak_adhesion_force_n": anomaly.get("peak_adhesion_force_n"),
                    "maximum_route_deviation_m": anomaly.get(
                        "maximum_absolute_route_lateral_offset_m"
                    ),
                    "effect_window_s": effect.get("window_duration_s"),
                    "progress_gain_lag_m": effect.get("progress_gain_lag_m"),
                    "max_tilt_increase_rad": effect.get("max_tilt_increase_rad"),
                    "min_base_height_drop_m": effect.get("min_base_height_drop_m"),
                    "total_attachment_cycles": last.get("total_attachment_cycles"),
                    "total_peel_cycles": last.get("total_peel_cycles"),
                    "total_tangential_work_j": last.get("total_tangential_work_j"),
                },
                "identity_passed": path.is_file()
                and manifest.get("development_only") is True
                and manifest.get("counts_as_a0_a7_evidence") is False
                and float(surface.get("parameters", {}).get("tangential_force_cap_n", -1.0))
                == float(case["tangential_force_cap_n"]),
            }
        )
    selected = sorted(row["tangential_force_cap_n"] for row in rows if row["selected"])
    integrity_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(record) for record in config["locked_files"]),
        "both_frozen_doses_present": sorted(row["tangential_force_cap_n"] for row in rows)
        == [18.0, 24.0],
        "identity_passed": all(row["identity_passed"] for row in rows),
    }
    audit = {
        "schema_version": "kinofail.o4-surface-v19-strength-calibration-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
        "rows": rows,
        "selected_tangential_force_cap_n": selected[0] if selected else None,
        "selection_passed": bool(selected),
        "passed": bool(selected) and all(integrity_checks.values()),
        "sealed": all(integrity_checks.values()),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "diagnosis": (
            "Both stronger lanes retained contact-aware peel events and route containment but "
            "fell at the same step. Because peel height/dwell also changed from v18, this is "
            "not a single-factor force-dose claim and no strength is selected."
        ),
        "next_gate": (
            "Freeze an action-conditioned train-only branch at a common post-attachment decision "
            "state: Continue versus Backstep, requiring pre-decision equality and a favorable "
            "Backstep recovery consequence before any new held-out scenes."
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": audit["passed"], "sealed": audit["sealed"], "selected": audit["selected_tangential_force_cap_n"]}, indent=2))
    return 0 if audit["sealed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
