from __future__ import annotations

import sys
from pathlib import Path

from scripts import extract_kinofail_confirmatory_t2_features_storage_v1 as storage
from scripts import run_kinofail_reconfirmation_scene_pipeline_v4 as pipeline


SCENE = "confirm_v2_production_scene_04"


def test_f19_home_and_data_corpus_are_exact_mirrors() -> None:
    logical = (
        pipeline.ROOT
        / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
    )
    _, physical, relative = storage.mirrored_storage_mapping(logical)
    assert relative == physical.relative_to(storage.DATA_REPO_ROOT)
    assert str(physical).startswith("/data/eureka/KinoVLA/")


def test_f19_rewrites_only_t2_entrypoint() -> None:
    command = [
        sys.executable,
        str(pipeline.FROZEN_T2),
        "--corpus",
        "corpus",
    ]
    rewritten = pipeline.rewrite_storage_command("t2_features", command)
    assert Path(rewritten[1]).resolve() == pipeline.STORAGE_T2.resolve()
    assert rewritten[2:] == command[2:]
    assert pipeline.rewrite_storage_command("scale_snapshots", command) == command


def test_f19_pruner_receives_physical_data_root() -> None:
    logical_root = (
        pipeline.ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
    )
    logical_shard = logical_root / SCENE
    command = [
        sys.executable,
        "pruner.py",
        "--corpus-root",
        str(logical_root),
        "--corpus-shard",
        str(logical_shard),
    ]
    rewritten = pipeline.rewrite_storage_command("seal_and_prune", command)
    assert Path(
        rewritten[rewritten.index("--corpus-root") + 1]
    ) == pipeline.DATA_CORPUS_ROOT
    assert Path(
        rewritten[rewritten.index("--corpus-shard") + 1]
    ) == pipeline.DATA_CORPUS_ROOT / SCENE


def test_f19_manifest_validates() -> None:
    amendment = pipeline._validate_f19()
    assert amendment["passed"] is True
    assert amendment["scientific_contract"]["feature_values_or_order_changed"] is False
