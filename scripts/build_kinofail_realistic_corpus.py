#!/usr/bin/env python3
"""Compile reproducible Kino-Fail pilot/full schedules and run design leakage gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kino_vla.data.realistic_benchmark import (  # noqa: E402
    audit_registry_bound_schedule,
    build_realistic_corpus,
    write_corpus_build,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full", "both"), default="both")
    parser.add_argument("--config", default="data/kinofail_realistic.yaml")
    parser.add_argument("--out", default="outputs/kinofail_realistic/design_v1")
    parser.add_argument(
        "--registry-audit",
        type=Path,
        default=REPO_ROOT / "outputs/kinofail_realistic/scene_registry/registry_audit.json",
    )
    parser.add_argument(
        "--require-registry-binding",
        action="store_true",
        help="Fail unless the schedule is fully bound to admitted physical scene entities.",
    )
    args = parser.parse_args()

    modes = ("pilot", "full") if args.mode == "both" else (args.mode,)
    failed = False
    for mode in modes:
        build = build_realistic_corpus(args.config, mode=mode)
        paths = write_corpus_build(build, Path(args.out))
        binding = None
        if args.registry_audit.is_file():
            registry_audit = json.loads(args.registry_audit.read_text(encoding="utf-8"))
            binding = audit_registry_bound_schedule(build.records, registry_audit)
            binding_path = Path(args.out) / f"{mode}_registry_binding_audit.json"
            binding_path.write_text(
                json.dumps(binding, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            paths["registry_binding_audit"] = str(binding_path)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "records": len(build.records),
                    "passed": build.audit["passed"],
                    "operator_material_nmi": build.audit["operator_material_normalized_mi"],
                    "condition_material_nmi": build.audit["condition_material_normalized_mi"],
                    "registry_bound": None if binding is None else binding["passed"],
                    "outputs": paths,
                },
                ensure_ascii=False,
            )
        )
        failed = failed or not bool(build.audit["passed"])
        if args.require_registry_binding:
            failed = failed or binding is None or not bool(binding["passed"])
    if failed:
        raise SystemExit("Kino-Fail realistic design gate failed")


if __name__ == "__main__":
    main()
