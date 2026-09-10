#!/usr/bin/env python3
"""Prepare frozen features for original Scale/T2 and combined F4m T3."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/prepare_kino_v4_confirmation_features_v1.py"
EXPECTED_PREDECESSOR_SHA256 = "279796aaa4c68d3ae2929689e0f0083334b55a305bf8af2d33bab47742f820aa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module() -> types.ModuleType:
    if sha256(PREDECESSOR) != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F4m feature predecessor hash drift")
    source = PREDECESSOR.read_text(encoding="utf-8")
    replacements = {
        'CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")\n': (
            'CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")\n'
            'T3_ORIGINAL_CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_t3_runin_f4e/corpus")\n'
            'T3_EXTENSION_CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_t3_extension_f4h/corpus")\n'
            'T3_CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_t3_combined_f4m/corpus")\n'
            'EXT_DESIGN = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design"\n'
        ),
        'F4 = ROOT / "outputs/freeze/kino_v4_confirmation_observations_f4/observation_seal.json"\n': (
            'F4 = ROOT / "outputs/freeze/kino_v4_confirmation_observations_f4m/observation_seal.json"\n'
        ),
        'DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_v4_confirmation_v1"\n': (
            'DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_v4_confirmation_v1_f4m"\n'
        ),
        '    all_t3_cases = _jsonl(DESIGN / "schedules/global/t3_cases.jsonl")\n': (
            '    all_t3_cases = (\n'
            '        _jsonl(DESIGN / "schedules/global/t3_cases.jsonl")\n'
            '        + _jsonl(EXT_DESIGN / "schedules/global/t3_cases.jsonl")\n'
            '    )\n'
        ),
        '    for scene in scenes:\n'
        '        all_t3_rows.extend(\n'
        '            _jsonl(DESIGN / "schedules/scenes" / scene / "c2_t3/schedule.jsonl")\n'
        '        )\n': (
            '    for scene in scenes:\n'
            '        all_t3_rows.extend(\n'
            '            _jsonl(DESIGN / "schedules/scenes" / scene / "c2_t3/schedule.jsonl")\n'
            '        )\n'
            '    extension_scenes = [\n'
            '        str(row["scene_id"]) for row in _json(EXT_DESIGN / "scene_registry.json")["scenes"]\n'
            '    ]\n'
            '    for scene in extension_scenes:\n'
            '        all_t3_rows.extend(\n'
            '            _jsonl(EXT_DESIGN / "schedules/scenes" / scene / "c2_t3/schedule.jsonl")\n'
            '        )\n'
        ),
        '        or (288 - len(t3_cases)) / 288 >= ATTRITION_LIMIT\n': (
            '        or (len(all_t3_cases) - len(t3_cases)) / len(all_t3_cases) >= ATTRITION_LIMIT\n'
        ),
        '            str(CORPUS),\n'
        '            "--output",\n'
        '            str(base),\n': (
            '            str(T3_CORPUS),\n'
            '            "--output",\n'
            '            str(base),\n'
        ),
        '        str(CORPUS),\n'
        '        "--output",\n'
        '        str(invariant),\n': (
            '        str(T3_CORPUS),\n'
            '        "--output",\n'
            '        str(invariant),\n'
        ),
        '        "planned_cases_per_cell": 288,\n': (
            '        "planned_cases_by_cell": {\n'
            '            "T2_vision_decisive": 288,\n'
            '            "T3_proprio_decisive": len(all_t3_cases),\n'
            '        },\n'
        ),
        '            "T3_proprio_decisive": (288 - len(t3_cases)) / 288,\n': (
            '            "T3_proprio_decisive": (len(all_t3_cases) - len(t3_cases)) / len(all_t3_cases),\n'
        ),
        '    output.mkdir(parents=True, exist_ok=False)\n'
        '    scale = _prepare_scale(output, seal, scenes)\n': (
            '    if T3_CORPUS.exists():\n'
            '        raise FileExistsError(T3_CORPUS)\n'
            '    T3_CORPUS.mkdir(parents=True, exist_ok=False)\n'
            '    extension_scene_ids = {\n'
            '        str(row["scene_id"]) for row in _json(EXT_DESIGN / "scene_registry.json")["scenes"]\n'
            '    }\n'
            '    for scene in scenes:\n'
            '        os.symlink(T3_ORIGINAL_CORPUS / scene, T3_CORPUS / scene, target_is_directory=True)\n'
            '    for scene in sorted(extension_scene_ids):\n'
            '        os.symlink(T3_EXTENSION_CORPUS / scene, T3_CORPUS / scene, target_is_directory=True)\n'
            '    output.mkdir(parents=True, exist_ok=False)\n'
            '    scale = _prepare_scale(output, seal, scenes)\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"F4m feature patch point is not unique: {old!r}")
        source = source.replace(old, new)
    result = types.ModuleType("kino_v4_confirmation_features_f4m")
    result.__file__ = str(Path(__file__).resolve())
    result.__package__ = "scripts"
    exec(compile(source, str(PREDECESSOR), "exec"), result.__dict__)
    return result


def main() -> int:
    return int(module().main())


if __name__ == "__main__":
    raise SystemExit(main())
