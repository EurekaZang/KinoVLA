#!/usr/bin/env python3
"""Bind independently frozen A1/A4/A5/A6 measurements to the scale-v8 A0 chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs/eval/realistic_a0_a7_v4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complete(value: dict) -> bool:
    return value.get("status") in {
        "confirmatory_complete",
        "confirmatory_passed",
        "publication_ready",
    }


def _rebind(source: Path, destination: Path, *, a0_bundle: str, a0_sha: str) -> None:
    value = copy.deepcopy(_json(source))
    old_bundle = value.get("a0_evidence_bundle_sha256")
    value["a0_evidence_bundle_sha256"] = a0_bundle
    if "a0_certificate_sha256" in value:
        value["a0_certificate_sha256"] = a0_sha
    value["evidence_rebinding"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": _sha(source),
        "source_a0_evidence_bundle_sha256": old_bundle,
        "destination_a0_evidence_bundle_sha256": a0_bundle,
        "policy": (
            "The independently frozen side-experiment measurements and efficacy verdict are "
            "unchanged; only their corpus-level provenance anchor is rebound to scale-v8 A0."
        ),
    }
    _write(destination, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v5",
    )
    parser.add_argument(
        "--c4-direct",
        type=Path,
        default=None,
        help=(
            "Optional direct C4 report. It must compare the deployed selective policy with "
            "always-safe; A4 recovery-minus-continue is deliberately not accepted as a proxy."
        ),
    )
    args = parser.parse_args()
    target_root = args.target_root.resolve()
    a0_path = target_root / "a0_certificate.json"
    a0 = _json(a0_path)
    if a0.get("status") != "publication_ready":
        raise RuntimeError("scale-v8 A0 is not publication-ready")
    a0_bundle = str(a0["a0_evidence_bundle_sha256"])
    a0_sha = _sha(a0_path)

    for name in ("a1_construct_validity.json", "a4_consequence.json", "a6_boundary.json"):
        _rebind(
            V4 / name,
            target_root / name,
            a0_bundle=a0_bundle,
            a0_sha=a0_sha,
        )

    a5_path = target_root / "a5_selective.json"
    a5 = _json(a5_path)
    lighting_path = V4 / "a5_lighting_heldout.json"
    realization_path = V4 / "a5_physical_realization_heldout.json"
    a4_path = target_root / "a4_consequence.json"
    lighting, realization, a4 = map(_json, (lighting_path, realization_path, a4_path))
    a5["updated_utc"] = datetime.now(UTC).isoformat()
    a5["heldout_lighting"] = {
        "artifact": str(lighting_path.relative_to(ROOT)),
        "sha256": _sha(lighting_path),
        "status": lighting.get("status"),
        "efficacy_or_integrity_passed": lighting.get("passed") is True,
        "valid_counterfactual_pairs": lighting.get("valid_counterfactual_pairs"),
        "excluded_scheduled_pairs": lighting.get("excluded_scheduled_pairs"),
        "summaries": lighting.get("summaries", {}),
        "paired_scene_cluster_statistics": lighting.get("paired_scene_cluster_statistics", {}),
    }
    a5["heldout_physical_realization"] = {
        "artifact": str(realization_path.relative_to(ROOT)),
        "sha256": _sha(realization_path),
        "status": realization.get("status"),
        "efficacy_or_integrity_passed": realization.get("passed") is True,
        "valid_counterfactual_pairs": realization.get("valid_counterfactual_pairs"),
        "excluded_scheduled_pairs": realization.get("excluded_scheduled_pairs"),
        "operators": realization.get("operators", []),
        "summaries": realization.get("summaries", {}),
        "scene_cluster_statistics": realization.get("scene_cluster_statistics", {}),
        "claim_boundary": realization.get("claim_boundary"),
    }
    a5["actual_action_consequence"] = {
        "artifact": str(a4_path.relative_to(ROOT)),
        "sha256": _sha(a4_path),
        "status": a4.get("status"),
        "efficacy_passed": a4.get("passed") is True,
        "overall_recovery_minus_continue_cost": a4.get("overall", {}).get(
            "paired_recovery_minus_continue_cost"
        ),
    }
    c4_path = (
        args.c4_direct.resolve()
        if args.c4_direct is not None
        else target_root / "a5_c4_direct.json"
    )
    c4 = _json(c4_path) if c4_path.exists() else None
    if c4 is None:
        a5["direct_selective_vs_always_safe"] = {
            "status": "not_run",
            "required_estimand": "selective_policy_minus_always_safe_policy",
            "a4_estimand_is_not_a_substitute": "recovery_action_minus_continue_action",
        }
    else:
        if c4.get("estimand") != "selective_policy_minus_always_safe_policy":
            raise RuntimeError(
                "direct C4 artifact has the wrong estimand: "
                f"{c4.get('estimand')!r}"
            )
        a5["direct_selective_vs_always_safe"] = {
            "artifact": str(c4_path.relative_to(ROOT)),
            "sha256": _sha(c4_path),
            "status": c4.get("status"),
            "passed": c4.get("passed") is True,
            "estimand": c4.get("estimand"),
            "primary_endpoint": c4.get("primary_endpoint"),
            "released_attribution_precision": c4.get(
                "released_attribution_precision"
            ),
        }
    acceptance = dict(a5.get("acceptance", {}))
    acceptance.pop("selective_consequence_available_after_A4", None)
    acceptance["lighting_heldout_axis_present"] = _complete(lighting)
    acceptance["physical_realization_heldout_axis_present"] = _complete(realization)
    acceptance["actual_action_consequence_available"] = _complete(a4)
    acceptance["direct_selective_vs_always_safe_evaluation_present"] = (
        c4 is not None and _complete(c4)
    )
    acceptance["selective_minus_always_safe_cluster_ci_upper_le_0"] = (
        c4 is not None
        and c4.get("acceptance", {}).get(
            "selective_minus_always_safe_cluster_ci_upper_le_0"
        )
        is True
    )
    acceptance["released_attribution_precision_ge_0_95"] = (
        c4 is not None
        and c4.get("acceptance", {}).get(
            "released_attribution_precision_ge_0_95"
        )
        is True
    )
    a5["acceptance"] = acceptance
    all_side_measurements_complete = all(_complete(value) for value in (lighting, realization, a4))
    a5["status"] = "confirmatory_complete" if all_side_measurements_complete else "partial_complete"
    a5["passed"] = all(value is True for value in acceptance.values())
    a5["claim_boundary"] = (
        "All frozen held-out axes and A4 actual-action side evidence are retained, including "
        "negative results. C4 requires a direct selective-policy versus always-safe-policy "
        "comparison. A4 recovery-minus-continue neither validates nor refutes that estimand."
        if c4 is None
        else str(c4.get("claim_boundary", "Direct C4 result is bound above."))
    )
    a5["side_evidence_binding"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "a0_evidence_bundle_sha256": a0_bundle,
        "lighting_sha256": _sha(lighting_path),
        "physical_realization_sha256": _sha(realization_path),
        "actual_action_sha256": _sha(a4_path),
        "direct_c4_sha256": _sha(c4_path) if c4 is not None else None,
    }
    _write(a5_path, a5)

    print(
        json.dumps(
            {
                "a0_evidence_bundle_sha256": a0_bundle,
                "rebound": ["A1", "A4", "A6"],
                "a5_status": a5["status"],
                "a5_passed": a5["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
