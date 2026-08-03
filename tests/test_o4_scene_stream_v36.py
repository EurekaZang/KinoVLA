from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v36 as audit


def test_v36_requires_six_scenes_and_four_families() -> None:
    assert audit._quota_contract(
        {"minimum_admitted_scenes": 6, "minimum_admitted_families": 4}
    )
    assert not audit._quota_contract(
        {"minimum_admitted_scenes": 5, "minimum_admitted_families": 4}
    )
    assert not audit._quota_contract(
        {"minimum_admitted_scenes": 6, "minimum_admitted_families": 3}
    )


def test_v36_has_distinct_development_freeze_status() -> None:
    assert "development" in audit.EXPECTED_FREEZE_STATUS
    assert "heldout" not in audit.EXPECTED_FREEZE_STATUS
