#!/usr/bin/env python3
"""Download the approved CC0 PBR subset, freeze hashes, and render an audit sheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kino_vla.sim.terrain_materials import (  # noqa: E402
    render_material_contact_sheet,
    sync_terrain_materials,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/kinofail_realistic.yaml")
    parser.add_argument("--out", default="outputs/assets/terrain_pbr_v1")
    parser.add_argument("--material", action="append", dest="materials")
    args = parser.parse_args()

    lock = sync_terrain_materials(
        args.config,
        asset_root=args.out,
        material_ids=args.materials,
    )
    sheet = render_material_contact_sheet(
        lock,
        asset_root=args.out,
        output_path=Path(args.out) / "terrain_material_contact_sheet.jpg",
    )
    print(
        json.dumps(
            {
                "passed": lock["audit"]["passed"],
                "materials": lock["audit"]["n_materials"],
                "lock": str(Path(args.out) / "terrain_assets.lock.json"),
                "contact_sheet": str(sheet),
            }
        )
    )
    if not lock["audit"]["passed"]:
        raise SystemExit("terrain PBR asset gate failed")


if __name__ == "__main__":
    main()
