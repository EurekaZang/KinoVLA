"""Deterministic design compiler and leakage gates for the realistic Kino-Fail corpus.

This module compiles the *experimental design* before an expensive Isaac replay is launched.
It deliberately keeps four variables separate:

``scene/geometry`` -> ``PBR appearance`` -> ``physics/operator`` -> ``sensor profile``.

Every anomalous episode is paired with a nominal counterfactual that has byte-identical scene,
geometry, appearance, camera, and seeds.  Consequently, a texture-only classifier cannot infer
whether the operator is active.  Material and scene families are also group-held-out across the
formal train/validation/test split.

The generated records are schedules, not simulated data.  They retain ``artifact_state=planned``
until a collector attaches runtime RGB/proprioception/telemetry artifacts and passes the runtime
quality gates.  This distinction prevents a paper manifest from overstating planned episodes as
completed evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kino_vla.utils.config import CONFIGS_DIR

SCHEMA_VERSION = "kinofail.realistic-design.v3"


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_hex(*parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = CONFIGS_DIR / resolved
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"top level of {resolved} must be a mapping")
    return value


@dataclass(frozen=True)
class CorpusBuild:
    """A compiled benchmark schedule plus its design-time audit."""

    mode: str
    records: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def audit_registry_bound_schedule(
    records: Sequence[Mapping[str, Any]], registry_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless every scheduled scene/operator is backed by an admitted entity.

    The design compiler is intentionally able to describe future data.  A *formal* schedule,
    however, must not turn those conceptual scene IDs into apparent collected evidence.  This
    audit joins every schedule row to a passed registry scene and checks per-scene operator
    capability rather than relying only on global coverage counts.
    """

    scheduled_scenes = {str(row["scene_family"]) for row in records}
    required_by_scene: dict[str, set[str]] = defaultdict(set)
    for row in records:
        required_by_scene[str(row["scene_family"])].add(str(row["target_operator"]))

    registry_rows = {
        str(row.get("scene_id")): row
        for row in registry_audit.get("scene_audits", [])
        if isinstance(row, Mapping)
    }
    validated_scenes = {
        scene_id for scene_id, row in registry_rows.items() if row.get("passed") is True
    }
    missing_scenes = sorted(scheduled_scenes - validated_scenes)
    capability_gaps: dict[str, list[str]] = {}
    for scene_id in sorted(scheduled_scenes & validated_scenes):
        admitted = {str(value) for value in registry_rows[scene_id].get("operator_capabilities", [])}
        missing = sorted(required_by_scene[scene_id] - admitted)
        if missing:
            capability_gaps[scene_id] = missing

    registry_declares_fully_bound = (
        registry_audit.get("schedule_binding", {}).get("fully_bound") is True
    )
    registry_publication_ready = registry_audit.get("publication_ready") is True
    checks = {
        "registry_audit_passed": registry_audit.get("passed") is True,
        "every_scheduled_scene_validated": not missing_scenes,
        "every_scene_admits_every_scheduled_operator": not missing_scenes
        and not capability_gaps,
        "registry_declares_schedule_fully_bound": registry_declares_fully_bound,
        "registry_publication_ready": registry_publication_ready,
    }
    return {
        "schema_version": "kinofail.realistic-registry-bound-schedule-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "scheduled_scene_count": len(scheduled_scenes),
        "validated_scheduled_scene_count": len(scheduled_scenes & validated_scenes),
        "missing_scenes": missing_scenes,
        "operator_capability_gaps": capability_gaps,
        "interpretation": (
            "A design schedule is not a formal collection schedule until every concrete scene "
            "and every scene/operator pairing is admitted by the registry."
        ),
    }


def _indexed_by_id(items: Sequence[Mapping[str, Any]], *, field: str = "id") -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for raw in items:
        item = dict(raw)
        item_id = str(item[field])
        if item_id in indexed:
            raise ValueError(f"duplicate {field} {item_id!r}")
        indexed[item_id] = item
    return indexed


def _scene_cells(spec: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    domains = spec["domains"]
    cells: list[dict[str, Any]] = []
    for domain_id, domain in domains.items():
        scenes = domain["scenes"]
        if mode == "pilot":
            scene_count = int(spec["corpus"].get("pilot_scene_families_per_domain", 1))
            selected = scenes[:scene_count]
        else:
            selected = scenes
        for scene in selected:
            cells.append(
                {
                    "domain": str(domain_id),
                    "scene_family": str(scene["id"]),
                    "scene_source": str(scene["source"]),
                    "split": "pilot" if mode == "pilot" else str(scene["split"]),
                }
            )
    return cells


def _material_candidates(
    materials: Sequence[Mapping[str, Any]], *, domain: str, split: str, mode: str
) -> list[dict[str, Any]]:
    candidates = [
        dict(material)
        for material in materials
        if domain in material["domains"] and (mode == "pilot" or material["split"] == split)
    ]
    if len(candidates) < 2:
        raise ValueError(
            f"domain={domain!r} split={split!r} needs >=2 material families, got {len(candidates)}"
        )
    return sorted(candidates, key=lambda value: str(value["id"]))


def _render_tier(value: int, tier_cfg: Mapping[str, int]) -> str:
    bucket = value % 100
    pbr_end = int(tier_cfg["structured_pbr_percent"])
    decal_end = pbr_end + int(tier_cfg["pbr_decal_percent"])
    if bucket < pbr_end:
        return "structured_pbr"
    if bucket < decal_end:
        return "pbr_plus_decal"
    return "generated_sequence"


def _unit_interval(*parts: object) -> float:
    return (_stable_u64(*parts) % 10_001) / 10_000.0


def _appearance_views(
    *,
    spec: Mapping[str, Any],
    materials: Sequence[Mapping[str, Any]],
    scene: Mapping[str, Any],
    realization_index: int,
    severity_index: int,
    seed_index: int,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    # The operator is intentionally absent from every hash below.  For the same design cell, all
    # 11 causes therefore receive the same appearance distribution exactly, rather than merely in
    # expectation.  This is the central texture-label leakage invariant.
    key = (
        scene["domain"],
        scene["scene_family"],
        realization_index,
        severity_index,
        seed_index,
        mode,
    )
    candidates = _material_candidates(
        materials, domain=scene["domain"], split=scene["split"], mode=mode
    )
    states = spec["appearance_states"][scene["domain"]]
    rotation_choices = tuple(float(value) for value in spec["visual_randomization"]["rotation_deg"])
    scale_lo, scale_hi = (float(v) for v in spec["visual_randomization"]["uv_scale_range"])
    texture_cfg = spec["texture_swap_randomization"]
    view_count = int(texture_cfg["views_per_physical_episode"])
    if view_count < 2:
        raise ValueError("texture-swap design requires at least two appearance views")
    offset_lo, offset_hi = (float(v) for v in texture_cfg["uv_offset_range"])
    brightness_lo, brightness_hi = (
        float(v) for v in texture_cfg["albedo_brightness_multiplier_range"]
    )
    normal_lo, normal_hi = (float(v) for v in texture_cfg["normal_strength_range"])
    roughness_lo, roughness_hi = (float(v) for v in texture_cfg["roughness_multiplier_range"])
    material_start = _stable_u64("material", *key) % len(candidates)
    state_start = _stable_u64("surface-state", *key) % len(states)
    views: list[dict[str, Any]] = []
    for view_index in range(view_count):
        # Cycling through the domain/split-local catalog guarantees that a texture-swap group
        # covers multiple material families whenever at least two are available.  The operator is
        # still absent from the key, so this intervention cannot install an operator shortcut.
        material = candidates[(material_start + view_index) % len(candidates)]
        state = states[(state_start + view_index) % len(states)]
        view_key = (*key, view_index)
        uv_scale = scale_lo + _unit_interval("uv-scale", *view_key) * (scale_hi - scale_lo)
        rotation = rotation_choices[_stable_u64("uv-rotation", *view_key) % len(rotation_choices)]
        uv_offset = [
            round(
                offset_lo + _unit_interval("uv-offset-u", *view_key) * (offset_hi - offset_lo),
                6,
            ),
            round(
                offset_lo + _unit_interval("uv-offset-v", *view_key) * (offset_hi - offset_lo),
                6,
            ),
        ]
        brightness = brightness_lo + _unit_interval("brightness", *view_key) * (
            brightness_hi - brightness_lo
        )
        normal_strength = normal_lo + _unit_interval("normal-strength", *view_key) * (
            normal_hi - normal_lo
        )
        roughness = roughness_lo + _unit_interval("roughness", *view_key) * (
            roughness_hi - roughness_lo
        )
        tier = _render_tier(_stable_u64("render-tier", *view_key), spec["render_tiers"])
        appearance_seed = _stable_u64("appearance-seed", *view_key) % (2**31 - 1)
        view_id = "primary" if view_index == 0 else f"swap_{view_index:02d}"
        appearance_payload = (
            material["id"],
            state,
            round(uv_scale, 6),
            rotation,
            uv_offset,
            round(brightness, 6),
            round(normal_strength, 6),
            round(roughness, 6),
            tier,
        )
        views.append(
            {
                "appearance_view_id": view_id,
                "appearance_view_index": view_index,
                "is_primary": view_index == 0,
                "material_family": str(material["id"]),
                "material_asset_id": str(material["source_asset_id"]),
                "material_semantics": str(material["semantic_family"]),
                "material_source": str(material["source"]),
                "material_license": str(material["license"]),
                "physical_size_m": list(material["physical_size_m"]),
                "surface_state": str(state),
                "uv_scale": round(uv_scale, 6),
                "uv_rotation_deg": rotation,
                "uv_offset": uv_offset,
                "albedo_brightness_multiplier": round(brightness, 6),
                "normal_strength": round(normal_strength, 6),
                "roughness_multiplier": round(roughness, 6),
                "triplanar": bool(spec["visual_randomization"]["triplanar"]),
                "render_tier": tier,
                "visual_intervention_only": True,
                "appearance_seed": appearance_seed,
                "appearance_id": "app_" + _stable_hex(*appearance_payload, length=20),
            }
        )
    return tuple(views)


def _record(
    *,
    benchmark_id: str,
    mode: str,
    scene: Mapping[str, Any],
    operator: Mapping[str, Any],
    realization: str,
    realization_index: int,
    severity: Mapping[str, Any],
    severity_index: int,
    seed_index: int,
    appearance_views: Sequence[Mapping[str, Any]],
    camera_profile: str,
    condition: str,
) -> dict[str, Any]:
    pair_key = (
        benchmark_id,
        mode,
        scene["domain"],
        scene["scene_family"],
        operator["id"],
        realization,
        severity["id"],
        seed_index,
    )
    pair_id = "cf_" + _stable_hex(*pair_key, length=20)
    operator_seed = _stable_u64("operator", *pair_key) % (2**31 - 1)
    scene_seed = _stable_u64("scene", scene["domain"], scene["scene_family"], seed_index) % (
        2**31 - 1
    )
    geometry_id = f"{scene['scene_family']}::{operator['geometry_profiles'][realization_index]}"
    active = condition == "anomaly"
    physics_parameters = dict(severity["parameters"] if active else operator["nominal"])
    if not active:
        # Some operators (notably O8 transparent obstacles) require visible geometry/material
        # to stay byte-identical while only physics is disabled. Explicit inheritance prevents
        # a nominal/anomaly RGB label leak without silently applying this rule to every operator.
        for field in operator.get("counterfactual_inherit", []):
            physics_parameters[str(field)] = severity["parameters"][str(field)]
    episode_id = pair_id + ("_anomaly" if active else "_nominal")
    texture_swap_group_id = "ts_" + _stable_hex(*pair_key, condition, length=20)
    if not appearance_views:
        raise ValueError("appearance_views cannot be empty")
    appearance = dict(appearance_views[0])
    output_stem = f"{scene['split']}/{scene['domain']}/{operator['id']}/{episode_id}"
    rgb_views = {
        str(view["appearance_view_id"]): (
            output_stem + "/rgb/"
            if bool(view["is_primary"])
            else output_stem + f"/rgb_views/{view['appearance_view_id']}/"
        )
        for view in appearance_views
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "design_mode": mode,
        "episode_id": episode_id,
        "counterfactual_group_id": pair_id,
        "texture_swap_group_id": texture_swap_group_id,
        "appearance_view_count": len(appearance_views),
        "appearance_views": [dict(view) for view in appearance_views],
        "condition": condition,
        "artifact_state": "planned",
        "evaluation_eligible": False,
        "split": scene["split"],
        "domain": scene["domain"],
        "scene_family": scene["scene_family"],
        "scene_source": scene["scene_source"],
        "geometry_id": geometry_id,
        "geometry_profile": operator["geometry_profiles"][realization_index],
        "target_operator": operator["id"],
        "active_operator": operator["id"] if active else None,
        "attribution_category": operator["category"] if active else "nominal",
        "physical_realization": realization,
        "severity_id": severity["id"],
        "severity_rank": int(severity["rank"]),
        "physics_parameters": physics_parameters,
        "material_family": appearance["material_family"],
        "material_asset_id": appearance["material_asset_id"],
        "material_semantics": appearance["material_semantics"],
        "material_source": appearance["material_source"],
        "material_license": appearance["material_license"],
        "material_physical_size_m": appearance["physical_size_m"],
        "appearance_id": appearance["appearance_id"],
        "surface_state": appearance["surface_state"],
        "uv_scale": appearance["uv_scale"],
        "uv_rotation_deg": appearance["uv_rotation_deg"],
        "uv_offset": appearance["uv_offset"],
        "albedo_brightness_multiplier": appearance["albedo_brightness_multiplier"],
        "normal_strength": appearance["normal_strength"],
        "roughness_multiplier": appearance["roughness_multiplier"],
        "triplanar": appearance["triplanar"],
        "render_tier": appearance["render_tier"],
        "camera_profile": camera_profile,
        "scene_seed": scene_seed,
        "operator_seed": operator_seed,
        "appearance_seed": appearance["appearance_seed"],
        "required_outputs": {
            "rgb": output_stem + "/rgb/",
            "rgb_views": rgb_views,
            "depth_optional": output_stem + "/depth/",
            "proprio": output_stem + "/proprio.npz",
            "telemetry": output_stem + "/privileged.jsonl",
            "episode_manifest": output_stem + "/manifest.json",
        },
    }


def build_realistic_corpus(
    config: str | Path | Mapping[str, Any] = "data/kinofail_realistic.yaml",
    *,
    mode: str = "pilot",
) -> CorpusBuild:
    """Compile the pilot (396 records) or full (5,940 records) benchmark schedule."""
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be 'pilot' or 'full'")
    spec = _load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
    benchmark_id = str(spec["benchmark_id"])
    operators = _indexed_by_id(spec["operators"])
    materials = [dict(value) for value in spec["materials"]]
    scenes = _scene_cells(spec, mode)
    seed_count = int(spec["corpus"][f"{mode}_seeds"])
    severity_count = int(spec["corpus"][f"{mode}_severities"])
    realization_count = 1 if mode == "pilot" else int(spec["corpus"]["full_realizations"])
    camera_profiles = tuple(str(value) for value in spec["camera_profiles"])
    records: list[dict[str, Any]] = []
    for scene in scenes:
        for operator in operators.values():
            severities = (
                operator["severities"][-severity_count:]
                if mode == "pilot"
                else operator["severities"][:severity_count]
            )
            realizations = operator["physical_realizations"][:realization_count]
            for realization_index, realization in enumerate(realizations):
                for severity_index, severity in enumerate(severities):
                    for seed_index in range(seed_count):
                        appearance_views = _appearance_views(
                            spec=spec,
                            materials=materials,
                            scene=scene,
                            realization_index=realization_index,
                            severity_index=severity_index,
                            seed_index=seed_index,
                            mode=mode,
                        )
                        # Include the experimental cell, not only the scene.  The pilot has one
                        # seed and one realization per scene; hashing only those fields can leave
                        # a camera profile completely absent by chance, invalidating A0/A5.
                        camera_key = _stable_u64(
                            "camera",
                            scene["scene_family"],
                            operator["id"],
                            severity["id"],
                            realization_index,
                            seed_index,
                        )
                        camera_profile = camera_profiles[camera_key % len(camera_profiles)]
                        for condition in ("anomaly", "nominal_counterfactual"):
                            records.append(
                                _record(
                                    benchmark_id=benchmark_id,
                                    mode=mode,
                                    scene=scene,
                                    operator=operator,
                                    realization=str(realization),
                                    realization_index=realization_index,
                                    severity=severity,
                                    severity_index=severity_index,
                                    seed_index=seed_index,
                                    appearance_views=appearance_views,
                                    camera_profile=camera_profile,
                                    condition=condition,
                                )
                            )
    audit = audit_realistic_corpus(records, config=spec, mode=mode)
    return CorpusBuild(mode=mode, records=tuple(records), audit=audit)


def _mutual_information(rows: Iterable[Mapping[str, Any]], x_key: str, y_key: str) -> float:
    pairs = [(str(row[x_key]), str(row[y_key])) for row in rows]
    if not pairs:
        return 0.0
    total = float(len(pairs))
    joint = Counter(pairs)
    x_counts = Counter(x for x, _ in pairs)
    y_counts = Counter(y for _, y in pairs)
    value = 0.0
    for (x_value, y_value), count in joint.items():
        pxy = count / total
        px = x_counts[x_value] / total
        py = y_counts[y_value] / total
        value += pxy * math.log(pxy / (px * py))
    return value


def _normalized_mi(rows: Iterable[Mapping[str, Any]], x_key: str, y_key: str) -> float:
    rows = list(rows)
    mi = _mutual_information(rows, x_key, y_key)

    def entropy(key: str) -> float:
        counts = Counter(str(row[key]) for row in rows)
        total = float(len(rows))
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    denominator = max(entropy(x_key), entropy(y_key))
    return 0.0 if denominator == 0.0 else mi / denominator


def audit_realistic_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Check count, counterfactual identity, split isolation, and appearance leakage."""
    expected = int(config["corpus"][f"expected_{mode}_records"])
    ids = [str(row["episode_id"]) for row in records]
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        pairs[str(row["counterfactual_group_id"])].append(row)
    identity_fields = (
        "domain",
        "scene_family",
        "geometry_id",
        "target_operator",
        "physical_realization",
        "severity_id",
        "material_family",
        "appearance_id",
        "surface_state",
        "uv_scale",
        "uv_rotation_deg",
        "camera_profile",
        "scene_seed",
        "operator_seed",
        "appearance_seed",
        "appearance_views",
    )
    malformed_pairs: list[str] = []
    for pair_id, rows in pairs.items():
        conditions = {str(row["condition"]) for row in rows}
        identical = all(
            len({json.dumps(row[field], sort_keys=True) for row in rows}) == 1
            for field in identity_fields
        )
        if len(rows) != 2 or conditions != {"anomaly", "nominal_counterfactual"} or not identical:
            malformed_pairs.append(pair_id)

    split_by_material: dict[str, set[str]] = defaultdict(set)
    split_by_scene: dict[str, set[str]] = defaultdict(set)
    for row in records:
        for view in row.get("appearance_views", []):
            split_by_material[str(view["material_family"])].add(str(row["split"]))
        split_by_scene[str(row["scene_family"])].add(str(row["split"]))
    material_leaks = {
        key: sorted(value) for key, value in split_by_material.items() if len(value) > 1
    }
    scene_leaks = {key: sorted(value) for key, value in split_by_scene.items() if len(value) > 1}

    malformed_texture_swap_groups: list[str] = []
    min_swap_material_families = math.inf
    expected_views = int(config["texture_swap_randomization"]["views_per_physical_episode"])
    min_group_materials = int(
        config["quality_gates"]["min_material_families_per_texture_swap_group"]
    )
    expanded_views: list[dict[str, Any]] = []
    for row in records:
        raw_views = row.get("appearance_views")
        views = list(raw_views) if isinstance(raw_views, list) else []
        view_ids = [str(view.get("appearance_view_id")) for view in views]
        appearance_ids = [str(view.get("appearance_id")) for view in views]
        material_families = {str(view.get("material_family")) for view in views}
        primary = [view for view in views if view.get("is_primary") is True]
        primary_matches = bool(primary) and all(
            row.get(field) == primary[0].get(field)
            for field in (
                "appearance_id",
                "material_family",
                "surface_state",
                "uv_scale",
                "uv_rotation_deg",
                "uv_offset",
                "albedo_brightness_multiplier",
                "normal_strength",
                "roughness_multiplier",
                "appearance_seed",
            )
        )
        valid = (
            len(views) == expected_views
            and len(set(view_ids)) == expected_views
            and len(set(appearance_ids)) == expected_views
            and len(primary) == 1
            and primary[0].get("appearance_view_id") == "primary"
            and primary_matches
            and len(material_families) >= min_group_materials
            and all(view.get("visual_intervention_only") is True for view in views)
        )
        if not valid:
            malformed_texture_swap_groups.append(str(row.get("texture_swap_group_id")))
        min_swap_material_families = min(min_swap_material_families, len(material_families))
        for view in views:
            expanded_views.append(
                {
                    "episode_id": row["episode_id"],
                    "texture_swap_group_id": row["texture_swap_group_id"],
                    "condition": row["condition"],
                    "target_operator": row["target_operator"],
                    "material_family": view["material_family"],
                    "material_license": view["material_license"],
                    "appearance_view_id": view["appearance_view_id"],
                    "render_tier": view["render_tier"],
                }
            )

    anomalous = [row for row in records if row["condition"] == "anomaly"]
    anomalous_views = [row for row in expanded_views if row["condition"] == "anomaly"]
    per_operator_materials: dict[str, set[str]] = defaultdict(set)
    for row in anomalous_views:
        per_operator_materials[str(row["target_operator"])].add(str(row["material_family"]))
    min_material_families = min(
        (len(value) for value in per_operator_materials.values()), default=0
    )
    operator_material_nmi = _normalized_mi(anomalous_views, "target_operator", "material_family")
    condition_material_nmi = _normalized_mi(expanded_views, "condition", "material_family")
    operators_by_material: dict[str, set[str]] = defaultdict(set)
    conditions_by_material: dict[str, set[str]] = defaultdict(set)
    for view in expanded_views:
        operators_by_material[str(view["material_family"])].add(str(view["target_operator"]))
        conditions_by_material[str(view["material_family"])].add(str(view["condition"]))
    min_operators_per_material = min(
        (len(values) for values in operators_by_material.values()), default=0
    )
    materials_missing_both_conditions = sorted(
        material
        for material, conditions in conditions_by_material.items()
        if conditions != {"anomaly", "nominal_counterfactual"}
    )
    license_violations = sorted(
        {
            str(row["material_family"])
            for row in expanded_views
            if str(row["material_license"]) not in {"CC0", "CC-BY-4.0"}
        }
    )
    tier_counts = Counter(str(row["render_tier"]) for row in anomalous_views)
    split_counts = Counter(str(row["split"]) for row in records)
    operator_counts = Counter(str(row["target_operator"]) for row in anomalous)
    scene_families_by_domain: dict[str, set[str]] = defaultdict(set)
    scene_families_by_operator: dict[str, set[str]] = defaultdict(set)
    for row in anomalous:
        scene_families_by_domain[str(row["domain"])].add(str(row["scene_family"]))
        scene_families_by_operator[str(row["target_operator"])].add(str(row["scene_family"]))
    min_scene_families_per_domain = min(
        (len(values) for values in scene_families_by_domain.values()), default=0
    )
    min_scene_families_per_operator = min(
        (len(values) for values in scene_families_by_operator.values()), default=0
    )
    issues: list[str] = []
    if len(records) != expected:
        issues.append("record_count_mismatch")
    if len(set(ids)) != len(ids):
        issues.append("duplicate_episode_ids")
    if malformed_pairs:
        issues.append("counterfactual_identity_failure")
    if malformed_texture_swap_groups:
        issues.append("texture_swap_contract_failure")
    if material_leaks:
        issues.append("material_split_leakage")
    if scene_leaks:
        issues.append("scene_split_leakage")
    if min_material_families < int(config["quality_gates"]["min_material_families_per_operator"]):
        issues.append("insufficient_material_diversity")
    if min_scene_families_per_domain < int(
        config["quality_gates"]["min_scene_families_per_domain"]
    ):
        issues.append("insufficient_scene_family_diversity")
    if operator_material_nmi > float(config["quality_gates"]["max_operator_material_nmi"]):
        issues.append("operator_material_dependence")
    if condition_material_nmi > float(config["quality_gates"]["max_condition_material_nmi"]):
        issues.append("condition_material_dependence")
    if min_operators_per_material < int(
        config["quality_gates"]["min_operator_labels_per_material"]
    ):
        issues.append("insufficient_operator_coverage_per_material")
    if materials_missing_both_conditions:
        issues.append("material_condition_coverage_failure")
    if license_violations:
        issues.append("material_license_violation")
    return {
        "passed": not issues,
        "issues": issues,
        "mode": mode,
        "schema_version": SCHEMA_VERSION,
        "expected_records": expected,
        "actual_records": len(records),
        "anomaly_records": len(anomalous),
        "counterfactual_pairs": len(pairs),
        "malformed_counterfactual_pairs": malformed_pairs[:20],
        "texture_swap_groups": len(records),
        "appearance_view_records": len(expanded_views),
        "appearance_views_per_episode": expected_views,
        "malformed_texture_swap_groups": malformed_texture_swap_groups[:20],
        "min_material_families_per_texture_swap_group": (
            0 if math.isinf(min_swap_material_families) else int(min_swap_material_families)
        ),
        "min_operator_labels_per_material": min_operators_per_material,
        "materials_missing_both_conditions": materials_missing_both_conditions,
        "unique_episode_ids": len(set(ids)),
        "material_split_leaks": material_leaks,
        "scene_split_leaks": scene_leaks,
        "operator_material_normalized_mi": operator_material_nmi,
        "condition_material_normalized_mi": condition_material_nmi,
        "min_material_families_per_operator": min_material_families,
        "scene_families_by_domain": {
            key: sorted(values) for key, values in sorted(scene_families_by_domain.items())
        },
        "min_scene_families_per_domain": min_scene_families_per_domain,
        "min_scene_families_per_operator": min_scene_families_per_operator,
        "license_violations": license_violations,
        "render_tier_anomaly_counts": dict(sorted(tier_counts.items())),
        "split_record_counts": dict(sorted(split_counts.items())),
        "anomaly_operator_counts": dict(sorted(operator_counts.items())),
    }


def write_corpus_build(build: CorpusBuild, output_dir: str | Path) -> dict[str, str]:
    """Write JSONL schedule, audit, and summary with stable SHA-256 provenance."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schedule = output / f"{build.mode}_schedule.jsonl"
    with schedule.open("w", encoding="utf-8") as stream:
        for row in build.records:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    appearance_schedule = output / f"{build.mode}_appearance_views.jsonl"
    with appearance_schedule.open("w", encoding="utf-8") as stream:
        for row in build.records:
            required = row["required_outputs"]["rgb_views"]
            for view in row["appearance_views"]:
                flattened = {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_id": row["benchmark_id"],
                    "design_mode": row["design_mode"],
                    "episode_id": row["episode_id"],
                    "counterfactual_group_id": row["counterfactual_group_id"],
                    "texture_swap_group_id": row["texture_swap_group_id"],
                    "condition": row["condition"],
                    "split": row["split"],
                    "domain": row["domain"],
                    "scene_family": row["scene_family"],
                    "target_operator": row["target_operator"],
                    "attribution_category": row["attribution_category"],
                    "physical_realization": row["physical_realization"],
                    "severity_id": row["severity_id"],
                    "camera_profile": row["camera_profile"],
                    "rgb_output": required[view["appearance_view_id"]],
                    "proprio_output": row["required_outputs"]["proprio"],
                    "telemetry_output": row["required_outputs"]["telemetry"],
                    **dict(view),
                }
                stream.write(json.dumps(flattened, sort_keys=True, ensure_ascii=False) + "\n")
    audit_path = output / f"{build.mode}_design_audit.json"
    audit_path.write_text(
        json.dumps(build.audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path = output / f"{build.mode}_summary.json"
    summary = {
        "mode": build.mode,
        "schema_version": SCHEMA_VERSION,
        "records": len(build.records),
        "appearance_view_records": int(build.audit["appearance_view_records"]),
        "artifact_state": "planned",
        "design_gate_passed": bool(build.audit["passed"]),
        "schedule_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
        "appearance_view_schedule_sha256": hashlib.sha256(
            appearance_schedule.read_bytes()
        ).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schedule": str(schedule),
        "appearance_views": str(appearance_schedule),
        "audit": str(audit_path),
        "summary": str(summary_path),
    }
