#!/usr/bin/env python3
"""Train-only Isaac calibration for contact-aware O4 surface adhesion v2."""

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


CUSTOM_FLAGS = {
    "--tangential-force-cap-n",
    "--normal-force-cap-n",
    "--peel-height-m",
    "--unload-steps-to-peel",
    "--reattach-cooldown-steps",
    "--surface-max-active-feet",
}


def _strip_custom_args(argv: list[str]) -> list[str]:
    result = [argv[0]]
    skip = False
    for item in argv[1:]:
        if skip:
            skip = False
            continue
        if item in CUSTOM_FLAGS:
            skip = True
            continue
        if any(item.startswith(f"{name}=") for name in CUSTOM_FLAGS):
            continue
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tangential-force-cap-n", type=float, required=True)
    parser.add_argument("--normal-force-cap-n", type=float, required=True)
    parser.add_argument("--peel-height-m", type=float, required=True)
    parser.add_argument("--unload-steps-to-peel", type=int, required=True)
    parser.add_argument("--reattach-cooldown-steps", type=int, required=True)
    parser.add_argument("--surface-max-active-feet", type=int, required=True)
    args, _ = parser.parse_known_args()

    import kino_vla.sim.adhesion as adhesion_module
    import kino_vla.sim.isaac_policy_backend as backend_module
    from kino_vla.sim.adhesion_v2 import SurfaceAdhesionConfig
    from kino_vla.sim.isaac_o4_v2_backend import IsaacPolicyBackendO4V2
    from scripts.isaac_collect_realistic_route_o4_pair_v16 import main as collect_v16

    real_config = adhesion_module.FootAdhesionConfig
    real_backend = backend_module.IsaacPolicyBackend

    def surface_config(*factory_args: object, **kwargs: object) -> SurfaceAdhesionConfig:
        if factory_args:
            raise TypeError("v18 surface config requires keyword construction")
        return SurfaceAdhesionConfig(
            region=kwargs["region"],  # type: ignore[arg-type]
            surface_z_m=float(kwargs["surface_z_m"]),
            attach_contact_force_n=float(kwargs["attach_contact_force_n"]),
            detach_contact_force_n=2.0,
            attach_height_tolerance_m=min(float(kwargs["attach_height_tolerance_m"]), 0.04),
            tangential_stiffness_n_per_m=float(kwargs["stiffness_xy_n_per_m"]),
            tangential_damping_ns_per_m=float(kwargs["damping_xy_ns_per_m"]),
            normal_stiffness_n_per_m=float(kwargs["stiffness_z_n_per_m"]),
            normal_damping_ns_per_m=float(kwargs["damping_z_ns_per_m"]),
            tangential_force_cap_n=args.tangential_force_cap_n,
            normal_force_cap_n=args.normal_force_cap_n,
            peel_height_m=args.peel_height_m,
            unload_steps_to_peel=args.unload_steps_to_peel,
            reattach_cooldown_steps=args.reattach_cooldown_steps,
            max_active_feet=args.surface_max_active_feet,
            progress_axis_xy=kwargs["progress_axis_xy"],  # type: ignore[arg-type]
        )

    original_argv = sys.argv
    adhesion_module.FootAdhesionConfig = surface_config  # type: ignore[assignment]
    backend_module.IsaacPolicyBackend = IsaacPolicyBackendO4V2
    try:
        sys.argv = _strip_custom_args(original_argv)
        v16_return_code = collect_v16()
    finally:
        sys.argv = original_argv
        adhesion_module.FootAdhesionConfig = real_config
        backend_module.IsaacPolicyBackend = real_backend

    path = args.out.resolve() / "pair_manifest.json"
    if not path.is_file():
        return 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    nominal = manifest["takes"]["nominal"]
    anomaly = manifest["takes"]["o4_adhesion"]
    consequence = manifest["paired_consequence"]
    telemetry_path = Path(anomaly["telemetry"])
    last_telemetry = {}
    if telemetry_path.is_file():
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            last_telemetry = json.loads(line)["adhesion"]
    checks = {
        "nominal_completed_without_fall": nominal["fell"] is False
        and nominal["steps"] == nominal["fixed_horizon_steps"],
        "surface_operator_mode_readback": last_telemetry.get("mode")
        == "contact_aware_anisotropic_surface_adhesion_v2",
        "surface_attachment_observed": int(anomaly["attachment_events"]) >= 1,
        "surface_peel_cycle_observed": int(anomaly["peel_events"]) >= 1,
        "anomaly_completed_without_fall": anomaly["fell"] is False
        and anomaly["steps"] == anomaly["fixed_horizon_steps"],
        "route_budget_preserved": float(anomaly["maximum_absolute_route_lateral_offset_m"])
        <= float(manifest["route_controller"]["maximum_route_deviation_m"]),
        "paired_effect_window_available": consequence.get("available") is True,
        "paired_locomotion_consequence": consequence.get("available") is True
        and float(consequence.get("window_duration_s", -1.0)) >= 1.0
        and float(consequence.get("progress_gain_lag_m", -1.0)) >= 0.08,
        "paired_posture_consequence": consequence.get("available") is True
        and (
            float(consequence.get("max_tilt_increase_rad", -1.0)) >= 0.10
            or float(consequence.get("min_base_height_drop_m", -1.0)) >= 0.03
        ),
        "both_sequences_visually_valid": nominal["valid_frame_fraction"] >= 0.90
        and anomaly["valid_frame_fraction"] >= 0.90,
    }
    manifest["schema_version"] = "kinofail.realistic-route-o4-surface-calibration.v18"
    manifest["passed"] = all(checks.values())
    manifest["development_only"] = True
    manifest["counts_as_a0_a7_evidence"] = False
    manifest["realistic_a0_a7_readiness"] = "0/8"
    manifest["v18_surface_calibration"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "subtype": "adhesive_contact",
        "wrapped_v16_return_code": v16_return_code,
        "parameters": {
            "tangential_force_cap_n": args.tangential_force_cap_n,
            "normal_force_cap_n": args.normal_force_cap_n,
            "peel_height_m": args.peel_height_m,
            "unload_steps_to_peel": args.unload_steps_to_peel,
            "reattach_cooldown_steps": args.reattach_cooldown_steps,
            "max_active_feet": args.surface_max_active_feet,
        },
        "last_telemetry": last_telemetry,
        "checks": checks,
        "wrapper_sha256": _sha256(Path(__file__).resolve()),
        "surface_model_sha256": _sha256(ROOT / "kino_vla/sim/adhesion_v2.py"),
        "backend_adapter_sha256": _sha256(
            ROOT / "kino_vla/sim/isaac_o4_v2_backend.py"
        ),
        "interpretation_policy": "train-only architecture calibration, never A0-A7 evidence",
    }
    manifest["dataset_status"] = "train_only_o4_surface_architecture_calibration"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "passed": manifest["passed"], "checks": checks}, indent=2))
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
