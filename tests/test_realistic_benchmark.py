from __future__ import annotations

from collections import defaultdict

from kino_vla.data.realistic_benchmark import (
    audit_registry_bound_schedule,
    build_realistic_corpus,
)


def test_pilot_and_full_counts_pass_design_gate() -> None:
    pilot = build_realistic_corpus(mode="pilot")
    full = build_realistic_corpus(mode="full")
    assert len(pilot.records) == 396
    assert len(full.records) == 5940
    assert pilot.audit["passed"], pilot.audit
    assert full.audit["passed"], full.audit


def test_pilot_uses_independent_scene_families_not_seed_repeats() -> None:
    pilot = build_realistic_corpus(mode="pilot")
    scenes_by_domain = defaultdict(set)
    scenes_by_operator = defaultdict(set)
    seeds_by_scene = defaultdict(set)
    for row in pilot.records:
        scenes_by_domain[row["domain"]].add(row["scene_family"])
        scenes_by_operator[row["target_operator"]].add(row["scene_family"])
        seeds_by_scene[row["scene_family"]].add(row["scene_seed"])
    assert all(len(scenes) == 3 for scenes in scenes_by_domain.values())
    assert all(len(scenes) == 9 for scenes in scenes_by_operator.values())
    assert all(len(seeds) == 1 for seeds in seeds_by_scene.values())
    assert pilot.audit["min_scene_families_per_domain"] == 3
    assert pilot.audit["min_scene_families_per_operator"] == 9


def test_every_anomaly_has_a_visual_identity_counterfactual() -> None:
    build = build_realistic_corpus(mode="pilot")
    pairs = defaultdict(list)
    for row in build.records:
        pairs[row["counterfactual_group_id"]].append(row)
    assert len(pairs) == 198
    for rows in pairs.values():
        assert {row["condition"] for row in rows} == {"anomaly", "nominal_counterfactual"}
        anomaly = next(row for row in rows if row["condition"] == "anomaly")
        nominal = next(row for row in rows if row["condition"] == "nominal_counterfactual")
        for field in (
            "scene_family",
            "geometry_id",
            "material_family",
            "appearance_id",
            "surface_state",
            "uv_scale",
            "uv_rotation_deg",
            "camera_profile",
            "scene_seed",
            "operator_seed",
            "appearance_seed",
        ):
            assert anomaly[field] == nominal[field]
        assert anomaly["active_operator"] == anomaly["target_operator"]
        assert nominal["active_operator"] is None


def test_o8_counterfactual_inherits_visible_panel_and_changes_only_collision() -> None:
    build = build_realistic_corpus(mode="pilot")
    pairs = defaultdict(list)
    for row in build.records:
        if row["target_operator"] == "O8_invisible_collider":
            pairs[row["counterfactual_group_id"]].append(row)
    assert pairs
    for rows in pairs.values():
        anomaly = next(row for row in rows if row["condition"] == "anomaly")
        nominal = next(row for row in rows if row["condition"] == "nominal_counterfactual")
        assert (
            anomaly["physics_parameters"]["obstacle_height_m"]
            == nominal["physics_parameters"]["obstacle_height_m"]
        )
        assert (
            anomaly["physics_parameters"]["optical_transmission"]
            == nominal["physics_parameters"]["optical_transmission"]
        )
        assert anomaly["physics_parameters"]["collision_enabled"] == 1.0
        assert nominal["physics_parameters"]["collision_enabled"] == 0.0


def test_full_split_holds_out_scene_and_material_families() -> None:
    build = build_realistic_corpus(mode="full")
    material_splits = defaultdict(set)
    scene_splits = defaultdict(set)
    for row in build.records:
        material_splits[row["material_family"]].add(row["split"])
        scene_splits[row["scene_family"]].add(row["split"])
    assert all(len(splits) == 1 for splits in material_splits.values())
    assert all(len(splits) == 1 for splits in scene_splits.values())
    assert set(build.audit["split_record_counts"]) == {"train", "val", "test"}


def test_texture_distribution_is_operator_and_condition_independent() -> None:
    build = build_realistic_corpus(mode="full")
    assert build.audit["operator_material_normalized_mi"] <= 0.02
    assert build.audit["condition_material_normalized_mi"] <= 0.001
    assert build.audit["min_material_families_per_operator"] >= 6


def test_each_physical_episode_has_synchronized_texture_swap_views() -> None:
    build = build_realistic_corpus(mode="pilot")
    assert build.audit["appearance_views_per_episode"] == 3
    assert build.audit["appearance_view_records"] == 3 * len(build.records)
    assert build.audit["min_material_families_per_texture_swap_group"] >= 2
    assert not build.audit["malformed_texture_swap_groups"]
    for row in build.records:
        views = row["appearance_views"]
        assert [view["appearance_view_id"] for view in views] == [
            "primary",
            "swap_01",
            "swap_02",
        ]
        assert len({view["appearance_id"] for view in views}) == 3
        assert len({view["material_family"] for view in views}) >= 2
        assert all(view["visual_intervention_only"] for view in views)
        assert row["appearance_id"] == views[0]["appearance_id"]
        assert row["required_outputs"]["rgb_views"]["primary"] == row["required_outputs"]["rgb"]


def test_texture_swap_assignment_is_identical_across_physics_counterfactual() -> None:
    build = build_realistic_corpus(mode="pilot")
    pairs = defaultdict(list)
    for row in build.records:
        pairs[row["counterfactual_group_id"]].append(row)
    for rows in pairs.values():
        anomaly = next(row for row in rows if row["condition"] == "anomaly")
        nominal = next(row for row in rows if row["condition"] == "nominal_counterfactual")
        assert anomaly["appearance_views"] == nominal["appearance_views"]
        assert anomaly["texture_swap_group_id"] != nominal["texture_swap_group_id"]


def test_design_is_deterministic_and_does_not_claim_runtime_completion() -> None:
    first = build_realistic_corpus(mode="pilot")
    second = build_realistic_corpus(mode="pilot")
    assert first.records == second.records
    assert all(row["artifact_state"] == "planned" for row in first.records)
    assert not any(row["evaluation_eligible"] for row in first.records)


def test_formal_schedule_requires_concrete_scene_and_operator_binding() -> None:
    build = build_realistic_corpus(mode="pilot")
    result = audit_registry_bound_schedule(
        build.records,
        {
            "passed": True,
            "publication_ready": False,
            "schedule_binding": {"fully_bound": False},
            "scene_audits": [
                {
                    "scene_id": "indoor_workstudio_01",
                    "passed": True,
                    "operator_capabilities": ["O4_tether"],
                }
            ],
        },
    )
    assert not result["passed"]
    assert len(result["missing_scenes"]) == 8
    assert result["operator_capability_gaps"]["indoor_workstudio_01"]


def test_fully_admitted_registry_can_bind_the_formal_schedule() -> None:
    build = build_realistic_corpus(mode="pilot")
    scenes = sorted({str(row["scene_family"]) for row in build.records})
    operators = sorted({str(row["target_operator"]) for row in build.records})
    result = audit_registry_bound_schedule(
        build.records,
        {
            "passed": True,
            "publication_ready": True,
            "schedule_binding": {"fully_bound": True},
            "scene_audits": [
                {
                    "scene_id": scene,
                    "passed": True,
                    "operator_capabilities": operators,
                }
                for scene in scenes
            ],
        },
    )
    assert result["passed"], result
    assert result["validated_scheduled_scene_count"] == 9
