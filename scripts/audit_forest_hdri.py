#!/usr/bin/env python3
"""Audit the frozen forest HDRI before authoring it into an Isaac scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.hdri_preflight import evaluate_hdri_array  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "outputs/assets/forest_hdri_polyhaven_v1/forest_hdri.lock.json",
    )
    parser.add_argument("--asset-id", default="monks_forest")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/assets/forest_hdri_polyhaven_v1/monks_forest/preflight.json",
    )
    args = parser.parse_args()

    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matches = [item for item in lock["assets"] if item["id"] == args.asset_id]
    if len(matches) != 1:
        raise ValueError(f"expected one HDRI asset {args.asset_id!r}")
    asset = matches[0]
    image_path = Path(asset["file"]["path"]).resolve()
    if _sha256(image_path) != asset["file"]["sha256"]:
        raise ValueError("HDRI differs from frozen lock")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode HDRI: {image_path}")
    thresholds = {
        "maximum_negative_fraction": 0.0,
        "minimum_luminance_std": 0.05,
        "minimum_dynamic_range_p99_p01": 20.0,
        "maximum_seam_relative_to_mean": 0.35,
    }
    audit = evaluate_hdri_array(
        image,
        expected_width=4096,
        expected_height=2048,
        thresholds=thresholds,
    )
    audit.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "asset_id": args.asset_id,
            "image": {"path": str(image_path), "sha256": _sha256(image_path)},
            "asset_lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
            "development_only": True,
            "counts_as_a0_a7_evidence": False,
        }
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite HDRI preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "audit": str(output)}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
