#!/usr/bin/env python3
"""Build the frozen 3,000-case confirmatory C2 base feature bundle.

This wrapper executes the audited C2-v3 feature implementation by exact hash
while replacing only its historical fixed-size assertions and artifact schema
names.  The visual/proprio/HOG computation and exact T2/T3 matching contracts
remain byte-for-byte inherited.  The post-interaction 80-D proprio contract is
applied subsequently by ``build_kinofail_realistic_c2_v5_features.py``.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
IMPLEMENTATION = (
    ROOT / "scripts/build_kinofail_realistic_c2_v3_confirmation_features.py"
)
EXPECTED_SHA256 = (
    "808c340ee38a2b6e5c4a74cf2105c2d452d9e94dc1ce5618f83aed1ba0f67611"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_SHA256:
        raise RuntimeError("confirmatory C2 feature dependency hash mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        (
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
        ): (
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
            '        "at_most_five_percent_case_attrition": (\n'
            '            t2_case_count >= 1425 and t3_case_count >= 1425\n'
            '        ),\n'
            '        "six_samples_per_valid_case": all(\n'
            '            sum(str(row["case_id"]) == case_id for row in rows) == 6\n'
            '            for case_id in {str(row["case_id"]) for row in rows}\n'
            '        ),\n'
            '        "four_class_support": {\n'
            '            str(row["attribution_category"]) for row in rows\n'
            '        } == {\n'
            '            "adhesion", "compliant_terrain",\n'
            '            "invisible_obstacle", "low_friction",\n'
            '        },\n'
            '        "fresh_scene_and_domain_support": (\n'
            '            1 <= len({str(row["scene_cluster"]) for row in rows}) <= 30\n'
            '            and len({str(row["domain"]) for row in rows}) == 3\n'
            '        ),\n'
        ),
        "            len(runtime_manifest_hashes) == 30\n": (
            "            len(runtime_manifest_hashes) == 2 * t3_case_count\n"
        ),
        '"kinofail.realistic-c2-v3-confirmation-features.v1"': (
            '"kinofail.unified-confirmatory-c2-base-features.v1"'
        ),
        '"cases_per_direction": 15,': (
            '"cases_per_direction": {"T2": t2_case_count, "T3": t3_case_count},'
        ),
        '"scene_clusters": 3,': (
            '"scene_clusters": len({str(row["scene_cluster"]) for row in rows}),'
        ),
        '"kinofail.realistic-c2-geometry-features.v3"': (
            '"kinofail.unified-confirmatory-c2-geometry-features.v1"'
        ),
        (
            '            path = (\n'
            '                c1_corpus\n'
            '                / str(row["case_id"])\n'
            '                / str(relative)\n'
            '            )\n'
        ): (
            '            path = (\n'
            '                Path(str(row["source_case_dir"]))\n'
            '                / str(relative)\n'
            '            )\n'
        ),
        (
            '    episode_dir = corpus / Path(\n'
            '        schedule["required_outputs"]["episode_manifest"]\n'
            '    ).parent\n'
        ): (
            '    episode_dir = (\n'
            '        corpus\n'
            '        / str(schedule["scene_cluster"])\n'
            '        / Path(schedule["required_outputs"]["episode_manifest"]).parent\n'
            '    )\n'
        ),
        (
            '                        "domain": str(case["domain"]),\n'
            '                        "split": "test",\n'
        ): (
            '                        "domain": str(case["domain"]),\n'
            '                        "cluster_material": str(case["cluster_material"]),\n'
            '                        "material_id": str(case["material_id"]),\n'
            '                        "split": "test",\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(
                f"confirmatory C2 feature patch point is not unique: {old!r}"
            )
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_confirmatory_c2_feature_implementation")
    module.__file__ = str(IMPLEMENTATION)
    module.__package__ = "scripts"
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return module


def main() -> int:
    return int(_load().main())


if __name__ == "__main__":
    raise SystemExit(main())
