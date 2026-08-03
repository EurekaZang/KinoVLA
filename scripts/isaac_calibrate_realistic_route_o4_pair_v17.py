#!/usr/bin/env python3
"""Train-only O4 force-cap calibration adapter around the frozen v16 collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_custom_args(argv: list[str]) -> list[str]:
    custom = {"--force-cap-n", "--endpoint-class"}
    result = [argv[0]]
    skip = False
    for item in argv[1:]:
        if skip:
            skip = False
            continue
        if item in custom:
            skip = True
            continue
        if any(item.startswith(f"{name}=") for name in custom):
            continue
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force-cap-n", type=float, required=True)
    parser.add_argument(
        "--endpoint-class", choices=("moderate_recoverable", "severe_terminal"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if not 0.0 < args.force_cap_n <= 35.0:
        parser.error("force cap must be in (0, 35] N for this frozen search")

    import kino_vla.sim.adhesion as adhesion_module
    from scripts.isaac_collect_realistic_route_o4_pair_v16 import main as collect_v16

    real_config = adhesion_module.FootAdhesionConfig

    def calibrated_config(*factory_args: object, **factory_kwargs: object) -> object:
        factory_kwargs["force_cap_n"] = float(args.force_cap_n)
        return real_config(*factory_args, **factory_kwargs)

    original_argv = sys.argv
    adhesion_module.FootAdhesionConfig = calibrated_config  # type: ignore[assignment]
    try:
        sys.argv = _strip_custom_args(original_argv)
        v16_return_code = collect_v16()
    finally:
        sys.argv = original_argv
        adhesion_module.FootAdhesionConfig = real_config

    manifest_path = args.out.resolve() / "pair_manifest.json"
    if not manifest_path.is_file():
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anomaly = manifest["takes"]["o4_adhesion"]
    nominal = manifest["takes"]["nominal"]
    consequence = manifest["paired_consequence"]
    recoverable = (
        nominal["fell"] is False
        and anomaly["fell"] is False
        and int(anomaly["peel_events"]) + int(anomaly["break_events"]) >= 1
        and consequence.get("available") is True
    )
    terminal = nominal["fell"] is False and anomaly["fell"] is True
    manifest["schema_version"] = "kinofail.realistic-route-o4-calibration.v17"
    manifest["development_only"] = True
    manifest["counts_as_a0_a7_evidence"] = False
    manifest["realistic_a0_a7_readiness"] = "0/8"
    manifest["v17_calibration"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "endpoint_class_requested": args.endpoint_class,
        "force_cap_n_requested": args.force_cap_n,
        "force_cap_n_readback": anomaly["peak_adhesion_force_n"],
        "wrapped_v16_return_code": v16_return_code,
        "moderate_recoverable_observed": recoverable,
        "severe_terminal_observed": terminal,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": _sha256(Path(__file__).resolve()),
        "wrapped_collector_sha256": _sha256(
            ROOT / "scripts/isaac_collect_realistic_route_o4_pair_v16.py"
        ),
        "interpretation_policy": (
            "Train-only dose selection. No row from this search may enter A0-A7 or held-out "
            "operator confirmation."
        ),
    }
    manifest["passed"] = bool(
        recoverable if args.endpoint_class == "moderate_recoverable" else terminal
    )
    manifest["dataset_status"] = "train_only_o4_endpoint_calibration"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "force_cap_n": args.force_cap_n,
                "endpoint_class": args.endpoint_class,
                "passed": manifest["passed"],
                "recoverable": recoverable,
                "terminal": terminal,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
