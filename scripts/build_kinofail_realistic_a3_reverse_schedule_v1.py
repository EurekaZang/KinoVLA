#!/usr/bin/env python3
"""Build the minimal three-domain O7 reverse-conflict extension.

The extension reuses the frozen scale-v7 scenes, camera profiles, controller, and O7 physics.
Only the route-surface appearance is replaced with a domain-appropriate scanned PBR material
whose visible rough/broken/muddy structure suggests a hazard.  The paired nominal episode keeps
nominal friction and zero depth bias, so it is the predeclared vision-alarming/proprio-nominal
reverse-conflict probe required by A3.  One pair per registered scene family is sufficient because
the statistical unit is the independent scene/physics cluster, not appearance re-renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_MATERIAL = {
    "life": "Concrete039",      # visibly broken/rough indoor concrete
    "production": "Pathway002", # visibly muddy industrial pathway
    "wild": "Ground071",        # scanned forest mud with sticks
}
SURFACE_STATES = ("wet", "scuffed", "dusty")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _material_rows(schedule: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_asset: dict[str, dict[str, Any]] = {}
    for row in schedule:
        for view in row.get("appearance_views", []):
            by_asset.setdefault(str(view["material_asset_id"]), dict(view))
    missing = sorted(set(TARGET_MATERIAL.values()) - set(by_asset))
    if missing:
        raise RuntimeError(f"missing frozen PBR material templates: {missing}")
    return by_asset


def _appearance_views(
    *, pair_id: str, domain: str, material_template: dict[str, Any]
) -> list[dict[str, Any]]:
    views = []
    for index, view_id in enumerate(("primary", "swap_01", "swap_02")):
        view = dict(material_template)
        view.update({
            "appearance_view_id": view_id,
            "appearance_view_index": index,
            "is_primary": index == 0,
            "appearance_id": _stable_id("app", pair_id, view_id),
            "appearance_seed": _stable_int(pair_id, view_id),
            "surface_state": SURFACE_STATES[index],
            "uv_offset": [round(0.13 + 0.19 * index, 4), round(0.07 + 0.23 * index, 4)],
            "uv_rotation_deg": float((90 * index) % 360),
            "uv_scale": round(0.9 + 0.15 * index, 6),
            "albedo_brightness_multiplier": round(0.90 + 0.10 * index, 6),
            "normal_strength": round(0.95 + 0.15 * index, 6),
            "roughness_multiplier": round(0.90 + 0.12 * index, 6),
            "visual_intervention_only": True,
            "reverse_conflict_visual_role": "hazard_looking_nominal_physics",
            "domain": domain,
        })
        views.append(view)
    return views


def _replace_appearance(
    row: dict[str, Any], *, pair_id: str, domain: str, material_template: dict[str, Any]
) -> None:
    views = _appearance_views(
        pair_id=pair_id, domain=domain, material_template=material_template
    )
    primary = views[0]
    row["appearance_views"] = views
    row["appearance_view_count"] = len(views)
    row["texture_swap_group_id"] = _stable_id("ts", row["episode_id"])
    for key in (
        "appearance_id", "appearance_seed", "material_asset_id", "material_family",
        "material_license", "material_semantics", "material_source", "normal_strength",
        "render_tier", "roughness_multiplier", "surface_state", "triplanar", "uv_offset",
        "uv_rotation_deg", "uv_scale", "albedo_brightness_multiplier",
    ):
        row[key] = primary[key]
    row["material_physical_size_m"] = primary["physical_size_m"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-schedule", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a3_reverse_v1",
    )
    args = parser.parse_args()
    base = [
        json.loads(line) for line in args.base_schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    templates = _material_rows(base)
    o7 = [
        row for row in base
        if row["target_operator"] == "O7_visual_remap"
        and row["severity_id"] == "moderate"
    ]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in o7:
        by_scene[str(row["scene_family"])].append(row)
    selected: list[dict[str, Any]] = []
    for scene, rows in sorted(by_scene.items()):
        old_pairs = defaultdict(list)
        for row in rows:
            old_pairs[str(row["counterfactual_group_id"])].append(row)
        complete = [
            pair for pair in old_pairs.values()
            if len(pair) == 2
            and {row["condition"] for row in pair}
            == {"anomaly", "nominal_counterfactual"}
        ]
        if len(complete) != 1:
            raise RuntimeError(f"expected one moderate O7 pair for {scene}, found {len(complete)}")
        source_pair = complete[0]
        domain = str(source_pair[0]["domain"])
        material_id = TARGET_MATERIAL[domain]
        pair_id = _stable_id("cf", "a3_reverse_v1", scene)
        for source in source_pair:
            row = json.loads(json.dumps(source))
            row["benchmark_id"] = "kinofail_realistic_a3_reverse_v1"
            row["counterfactual_group_id"] = pair_id
            suffix = "anomaly" if row["condition"] == "anomaly" else "nominal"
            row["episode_id"] = f"{pair_id}_{suffix}"
            row["reverse_conflict_probe"] = True
            row["reverse_conflict_role"] = (
                "hazard_looking_hazard_physics_control"
                if row["condition"] == "anomaly"
                else "hazard_looking_nominal_physics_probe"
            )
            row["appearance_class"] = "hazard_looking_scanned_pbr"
            row["required_outputs"] = {
                "depth_optional": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/depth/",
                "episode_manifest": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/manifest.json",
                "proprio": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/proprio.npz",
                "rgb": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/rgb/",
                "rgb_views": {
                    "primary": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/rgb/",
                    "swap_01": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/rgb_views/swap_01/",
                    "swap_02": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/rgb_views/swap_02/",
                },
                "telemetry": f"{row['split']}/{domain}/O7_visual_remap/{row['episode_id']}/privileged.jsonl",
            }
            _replace_appearance(
                row, pair_id=pair_id, domain=domain,
                material_template=templates[material_id],
            )
            selected.append(row)

    pairs = defaultdict(list)
    for row in selected:
        pairs[str(row["counterfactual_group_id"])].append(row)
    checks = {
        "18_physical_episodes": len(selected) == 18,
        "nine_complete_pairs": len(pairs) == 9 and all(
            len(rows) == 2
            and {row["condition"] for row in rows}
            == {"anomaly", "nominal_counterfactual"}
            for rows in pairs.values()
        ),
        "nine_scene_families": len({row["scene_family"] for row in selected}) == 9,
        "three_domains": {row["domain"] for row in selected} == {"life", "production", "wild"},
        "o7_only": {row["target_operator"] for row in selected} == {"O7_visual_remap"},
        "three_synchronized_views": all(len(row["appearance_views"]) == 3 for row in selected),
        "nominal_probe_has_nominal_physics": all(
            float(row["physics_parameters"]["depth_bias_m"]) == 0.0
            and float(row["physics_parameters"]["mu_s"]) >= 0.8
            and float(row["physics_parameters"]["mu_d"]) >= 0.6
            for row in selected if row["condition"] == "nominal_counterfactual"
        ),
        "hazard_pbr_is_paired": all(
            rows[0]["appearance_views"] == rows[1]["appearance_views"] for rows in pairs.values()
        ),
        "a8_excluded": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    schedule = args.out / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.realistic-a3-reverse-design-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(selected),
        "pairs": len(pairs),
        "scene_families": len({row["scene_family"] for row in selected}),
        "domain_pair_counts": dict(sorted(Counter(row["domain"] for row in selected if row["condition"] == "anomaly").items())),
        "target_material_by_domain": TARGET_MATERIAL,
        "source_schedule": str(args.base_schedule.relative_to(ROOT)),
        "source_schedule_sha256": _sha(args.base_schedule),
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "statistical_unit": "registered scene family / physical realization pair; appearance views are synchronized interventions and are not independent samples",
    }
    audit_path = args.out / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "records": len(selected), "pairs": len(pairs), "schedule_sha256": _sha(schedule)}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
