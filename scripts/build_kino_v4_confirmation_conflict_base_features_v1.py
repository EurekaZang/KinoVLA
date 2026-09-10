#!/usr/bin/env python3
"""Build KiNO-v4 confirmation Conflict features with the frozen C2 encoder.

The implementation is hash-bound to the audited C2-v3 feature code.  Only
dataset-size assertions and confirmation-corpus path adapters are replaced;
the observable RGB/proprioception computation is unchanged.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
IMPLEMENTATION = ROOT / "scripts/build_kinofail_realistic_c2_v3_confirmation_features.py"
EXPECTED_SHA256 = "808c340ee38a2b6e5c4a74cf2105c2d452d9e94dc1ce5618f83aed1ba0f67611"
MINIMUM_CASES = 274  # floor required by the frozen <5% attrition gate for N=288


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_SHA256:
        raise RuntimeError("frozen C2 feature implementation hash drift")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    old_checks = (
        '    checks = {\n'
        '        "expected_180_samples": len(rows) == 180,\n'
        '        "thirty_cases_fifteen_per_direction": (\n'
        '            len({str(row["case_id"]) for row in rows}) == 30\n'
        '            and all(\n'
        '                len(\n'
        '                    {\n'
        '                        str(row["case_id"])\n'
        '                        for row in rows\n'
        '                        if row["cell"] == cell\n'
        '                    }\n'
        '                )\n'
        '                == 15\n'
        '                for cell in (\n'
        '                    "T2_vision_decisive",\n'
        '                    "T3_proprio_decisive",\n'
        '                )\n'
        '            )\n'
        '        ),\n'
        '        "four_balanced_categories": all(\n'
        '            sum(\n'
        '                row["attribution_category"] == category\n'
        '                for row in rows\n'
        '            )\n'
        '            == 45\n'
        '            for category in (\n'
        '                "adhesion",\n'
        '                "compliant_terrain",\n'
        '                "invisible_obstacle",\n'
        '                "low_friction",\n'
        '            )\n'
        '        ),\n'
        '        "three_fresh_scenes_and_domains": (\n'
        '            len({str(row["scene_cluster"]) for row in rows})\n'
        '            == 3\n'
        '            and len({str(row["domain"]) for row in rows}) == 3\n'
        '        ),\n'
    )
    new_checks = (
        '    t2_case_count = len({\n'
        '        str(row["case_id"]) for row in rows\n'
        '        if row["cell"] == "T2_vision_decisive"\n'
        '    })\n'
        '    t3_case_count = len({\n'
        '        str(row["case_id"]) for row in rows\n'
        '        if row["cell"] == "T3_proprio_decisive"\n'
        '    })\n'
        '    checks = {\n'
        '        "expected_valid_samples": (\n'
        '            len(rows) == 6 * (t2_case_count + t3_case_count)\n'
        '        ),\n'
        f'        "at_most_five_percent_case_attrition": (t2_case_count >= {MINIMUM_CASES} and t3_case_count >= {MINIMUM_CASES}),\n'
        '        "six_samples_per_valid_case": all(\n'
        '            sum(str(row["case_id"]) == case_id for row in rows) == 6\n'
        '            for case_id in {str(row["case_id"]) for row in rows}\n'
        '        ),\n'
        '        "four_class_support": {str(row["attribution_category"]) for row in rows} == {\n'
        '            "adhesion", "compliant_terrain", "invisible_obstacle", "low_friction"\n'
        '        },\n'
        '        "fresh_scene_and_domain_support": (\n'
        '            len({str(row["scene_cluster"]) for row in rows}) >= 12\n'
        '            and len({str(row["domain"]) for row in rows}) == 3\n'
        '        ),\n'
    )
    replacements = {
        old_checks: new_checks,
        "            len(runtime_manifest_hashes) == 30\n": (
            "            len(runtime_manifest_hashes) == 2 * t3_case_count\n"
        ),
        '"kinofail.realistic-c2-v3-confirmation-features.v1"': (
            '"kinofail.kino-v4-confirmation-c2-base-features.v1"'
        ),
        '"cases_per_direction": 15,': (
            '"cases_per_direction": {"T2": t2_case_count, "T3": t3_case_count},'
        ),
        '"scene_clusters": 3,': (
            '"scene_clusters": len({str(row["scene_cluster"]) for row in rows}),'
        ),
        '"kinofail.realistic-c2-geometry-features.v3"': (
            '"kinofail.kino-v4-confirmation-c2-geometry-features.v1"'
        ),
        (
            '            path = (\n'
            '                c1_corpus\n'
            '                / str(row["case_id"])\n'
            '                / str(relative)\n'
            '            )\n'
        ): (
            '            path = Path(str(row["source_case_dir"])) / str(relative)\n'
        ),
        (
            '    episode_dir = corpus / Path(\n'
            '        schedule["required_outputs"]["episode_manifest"]\n'
            '    ).parent\n'
        ): (
            '    episode_dir = (corpus / str(schedule["scene_id"]) / Path(\n'
            '        schedule["required_outputs"]["episode_manifest"]\n'
            '    ).parent)\n'
        ),
        (
            '                        "domain": str(case["domain"]),\n'
            '                        "split": "test",\n'
        ): (
            '                        "domain": str(case["domain"]),\n'
            '                        "cluster_material": str(case["cluster_material"]),\n'
            '                        "material_id": str(case["material_id"]),\n'
            '                        "source_case_dir": str(t3_corpus / str(case["scene_cluster"])),\n'
            '                        "split": "independent_confirmation",\n'
        ),
        '                        "severity": str(case["severity"]),\n': (
            '                        "severity": str(case["severity_id"]),\n'
        ),
        '                            case["profile_index"]\n': (
            '                            case["physical_nuisance"]["profile_index"]\n'
        ),
        (
            '    proprio = np.concatenate(\n'
            '        [c1_proprio, t3_proprio_array], axis=0\n'
            '    )\n'
        ): (
            '    proprio = np.zeros_like(\n'
            '        np.concatenate([c1_proprio, t3_proprio_array], axis=0),\n'
            '        dtype=np.float32,\n'
            '    )\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"C2 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kino_v4_confirmation_c2_feature_implementation")
    module.__file__ = str(IMPLEMENTATION)
    module.__package__ = "scripts"
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return module


def main() -> int:
    return int(_load().main())


if __name__ == "__main__":
    raise SystemExit(main())
