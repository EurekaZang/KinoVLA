"""Public KINO names preserve the existing evaluation implementation and storage."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_one_canonical_file():
    canonical = ROOT / "benchmarks/KINO-Fail/evaluation.json"
    compatibility = ROOT / "configs/eval/publication_evaluation.json"
    assert canonical.samefile(compatibility)
    evaluator = load_script("evaluate_kino_fail.py")
    assert evaluator.REGISTRY == canonical
    assert evaluator.load_registry() == json.loads(compatibility.read_text())


def test_old_evaluation_entrypoint_resolves_to_the_same_code():
    assert (ROOT / "scripts/evaluate_publication.py").samefile(
        ROOT / "scripts/evaluate_kino_fail.py"
    )


def test_kino_cli_name_selects_the_registered_gmu(monkeypatch):
    evaluator = load_script("evaluate_kino_fail.py")
    observed = []
    monkeypatch.setattr(
        evaluator, "check_paths", lambda reg, methods, root: observed.extend(methods)
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_kino_fail.py", "--method", "kino", "--check-only"])
    assert evaluator.main() == 0
    assert observed == ["gmu"]


def test_storage_aliases_are_idempotent(tmp_path):
    setup = load_script("setup_kino_paths.py")
    source = tmp_path / "original"
    source.mkdir()
    alias = tmp_path / "KINO-Fail/features"
    links = {alias: source}
    setup.apply_links(links)
    setup.apply_links(links)
    assert source.is_dir() and alias.is_symlink() and alias.samefile(source)


def test_alias_collision_fails_before_creating_any_links(tmp_path):
    setup = load_script("setup_kino_paths.py")
    source = tmp_path / "original"
    source.mkdir()
    collision = tmp_path / "occupied"
    collision.mkdir()
    alias = tmp_path / "not_created"
    with pytest.raises(FileExistsError):
        setup.apply_links({alias: source, collision: source})
    assert not alias.exists()
    assert collision.is_dir() and not collision.is_symlink()


def test_public_model_is_the_registered_class():
    pytest.importorskip("torch")
    from kino import KINO, FusionDimensions
    from kino_vla.eval.known_multimodal_fusions import (
        FusionDimensions as RegisteredDimensions,
    )
    from kino_vla.eval.known_multimodal_fusions import GatedMultimodalUnit

    assert KINO is GatedMultimodalUnit
    assert FusionDimensions is RegisteredDimensions
