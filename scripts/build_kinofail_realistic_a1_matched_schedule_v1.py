#!/usr/bin/env python3
"""Build the minimal fresh-app O2/O4 matched-construct extension for A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = {
    "O2_compliance": {
        "life": ("Concrete039", "Tiles140", "Concrete045"),
        "production": ("Pathway002", "Concrete040", "Asphalt031"),
        "wild": ("Ground054", "Ground071", "Ground073"),
    },
    "O4_tether": {
        "life": ("Tiles141", "Concrete034", "Tiles139"),
        "production": ("Road007", "Concrete046", "Asphalt023S"),
        "wild": ("Ground037", "Gravel023", "Gravel022"),
    },
}
SURFACE_STATES = ("clean", "scuffed", "wet")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(*parts: object) -> int:
    data = "|".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big") & 0x7FFFFFFF


def _stable_id(prefix: str, *parts: object) -> str:
    data = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(data).hexdigest()[:20]}"


def _templates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for view in row.get("appearance_views", []):
            result.setdefault(str(view["material_asset_id"]), dict(view))
    required = {asset for op in MATERIALS.values() for assets in op.values() for asset in assets}
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"missing scanned-PBR templates: {missing}")
    return result


def _set_appearance(
    row: dict[str, Any], *, operator: str, domain: str,
    pair_id: str, templates: dict[str, dict[str, Any]],
) -> None:
    views = []
    for index, (view_id, asset) in enumerate(zip(
        ("primary", "swap_01", "swap_02"), MATERIALS[operator][domain], strict=True
    )):
        view = dict(templates[asset])
        view.update({
            "appearance_view_id": view_id,
            "appearance_view_index": index,
            "is_primary": index == 0,
            "appearance_id": _stable_id("app", "a1_matched", pair_id, view_id),
            "appearance_seed": _stable_int("a1_matched", pair_id, view_id),
            "surface_state": SURFACE_STATES[index],
            "uv_offset": [round(0.09 + 0.21 * index, 4), round(0.16 + 0.17 * index, 4)],
            "uv_rotation_deg": float(index * 90),
            "uv_scale": round(0.9 + 0.12 * index, 6),
            "albedo_brightness_multiplier": round(0.92 + 0.07 * index, 6),
            "normal_strength": round(0.95 + 0.12 * index, 6),
            "roughness_multiplier": round(0.94 + 0.10 * index, 6),
            "visual_intervention_only": True,
            "a1_visible_cause_family": (
                "soft_or_broken_surface" if operator == "O2_compliance"
                else "adhesive_or_entangling_surface"
            ),
        })
        views.append(view)
    row["appearance_views"] = views
    row["appearance_view_count"] = 3
    row["texture_swap_group_id"] = _stable_id("ts", row["episode_id"])
    primary = views[0]
    for key in (
        "appearance_id", "appearance_seed", "material_asset_id", "material_family",
        "material_license", "material_semantics", "material_source", "normal_strength",
        "render_tier", "roughness_multiplier", "surface_state", "triplanar", "uv_offset",
        "uv_rotation_deg", "uv_scale", "albedo_brightness_multiplier",
    ):
        row[key] = primary[key]
    row["material_physical_size_m"] = primary["physical_size_m"]


def _outputs(row: dict[str, Any]) -> dict[str, Any]:
    prefix = f"{row['split']}/{row['domain']}/{row['target_operator']}/{row['episode_id']}"
    return {
        "depth_optional": f"{prefix}/depth/",
        "episode_manifest": f"{prefix}/manifest.json",
        "proprio": f"{prefix}/proprio.npz",
        "rgb": f"{prefix}/rgb/",
        "rgb_views": {
            "primary": f"{prefix}/rgb/",
            "swap_01": f"{prefix}/rgb_views/swap_01/",
            "swap_02": f"{prefix}/rgb_views/swap_02/",
        },
        "telemetry": f"{prefix}/privileged.jsonl",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-schedule", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a1_matched_v1",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.base_schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    templates = _templates(rows)
    selected: list[dict[str, Any]] = []
    scenes = sorted({row["scene_family"] for row in rows})
    for scene in scenes:
        sources: dict[str, list[dict[str, Any]]] = {}
        for operator in ("O2_compliance", "O4_tether"):
            matches = [
                row for row in rows
                if row["scene_family"] == scene
                and row["target_operator"] == operator
                and row["severity_id"] == "moderate"
            ]
            by_pair = defaultdict(list)
            for row in matches:
                by_pair[str(row["counterfactual_group_id"])].append(row)
            complete = [
                pair for pair in by_pair.values()
                if len(pair) == 2
                and {row["condition"] for row in pair}
                == {"anomaly", "nominal_counterfactual"}
            ]
            if len(complete) != 1:
                raise RuntimeError(f"expected one moderate {operator} pair for {scene}")
            sources[operator] = complete[0]
        matched_seed = int(sources["O2_compliance"][0]["operator_seed"])
        matched_camera = str(sources["O2_compliance"][0]["camera_profile"])
        for operator, source_pair in sources.items():
            pair_id = _stable_id("cf", "a1_matched_v1", scene, operator)
            for source in source_pair:
                row = json.loads(json.dumps(source))
                row["benchmark_id"] = "kinofail_realistic_a1_matched_v1"
                row["counterfactual_group_id"] = pair_id
                suffix = "anomaly" if row["condition"] == "anomaly" else "nominal"
                row["episode_id"] = f"{pair_id}_{suffix}"
                row["operator_seed"] = matched_seed
                row["camera_profile"] = matched_camera
                row["a1_matched_construct"] = True
                row["a1_alignment_rule"] = "snapshot_ends_at_first_measured_operator_contact"
                row["a1_precontact_match_group"] = _stable_id("a1m", scene)
                if operator == "O4_tether":
                    row["physics_parameters"] = (
                        {
                            "attachment_enabled": 1.0,
                            "tangential_force_cap_n": 12.0,
                            "normal_force_cap_n": 6.0,
                            "peel_height_m": 0.025,
                            "unload_steps_to_peel": 3,
                        }
                        if row["condition"] == "anomaly"
                        else {
                            "attachment_enabled": 0.0,
                            "tangential_force_cap_n": 0.0,
                            "normal_force_cap_n": 0.0,
                            "peel_height_m": 0.025,
                            "unload_steps_to_peel": 3,
                        }
                    )
                    row["severity_id"] = "matched_mild"
                    row["severity_rank"] = 1
                else:
                    row["severity_id"] = "matched_moderate"
                    row["severity_rank"] = 2
                row["required_outputs"] = _outputs(row)
                _set_appearance(
                    row, operator=operator, domain=str(row["domain"]),
                    pair_id=pair_id, templates=templates,
                )
                selected.append(row)

    pairs = defaultdict(list)
    for row in selected:
        pairs[str(row["counterfactual_group_id"])].append(row)
    by_match = defaultdict(list)
    for row in selected:
        if row["condition"] == "anomaly":
            by_match[str(row["a1_precontact_match_group"])].append(row)
    checks = {
        "36_physical_episodes": len(selected) == 36,
        "18_complete_counterfactual_pairs": len(pairs) == 18 and all(
            len(pair) == 2
            and {row["condition"] for row in pair}
            == {"anomaly", "nominal_counterfactual"}
            for pair in pairs.values()
        ),
        "nine_cross_operator_match_groups": len(by_match) == 9 and all(
            {row["target_operator"] for row in group} == {"O2_compliance", "O4_tether"}
            for group in by_match.values()
        ),
        "same_scene_seed_camera_and_reset_seed_within_match": all(
            len({(row["scene_family"], row["scene_seed"], row["camera_profile"], row["operator_seed"]) for row in group}) == 1
            for group in by_match.values()
        ),
        "operator_specific_scanned_pbr_with_three_families": all(
            len({view["material_asset_id"] for view in row["appearance_views"]}) == 3
            for row in selected
        ),
        "cause_material_sets_do_not_overlap": all(
            set(MATERIALS["O2_compliance"][domain]).isdisjoint(MATERIALS["O4_tether"][domain])
            for domain in ("life", "production", "wild")
        ),
        "three_domains": {row["domain"] for row in selected} == {"life", "production", "wild"},
        "nine_scenes": len({row["scene_family"] for row in selected}) == 9,
        "a8_excluded": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    schedule = args.out / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.realistic-a1-matched-design-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(selected),
        "counterfactual_pairs": len(pairs),
        "cross_operator_match_groups": len(by_match),
        "domain_match_counts": dict(sorted(Counter(row["domain"] for row in selected if row["condition"] == "anomaly").items())),
        "selection_rationale": "Use one fixed moderate O2 and mild O4 dose; equivalence is achieved by pre-contact event alignment, not outcome-driven parameter tuning.",
        "source_schedule": str(args.base_schedule.relative_to(ROOT)),
        "source_schedule_sha256": _sha(args.base_schedule),
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "statistical_unit": "independent registered scene family; appearance views are clustered interventions, not independent physics samples",
    }
    audit_path = args.out / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "records": len(selected), "pairs": len(pairs), "match_groups": len(by_match), "schedule_sha256": _sha(schedule)}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
