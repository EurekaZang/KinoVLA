from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v34 as audit


def test_v34_requires_six_scenes_and_four_families() -> None:
    assert audit._quota_contract(
        {"minimum_admitted_scenes": 6, "minimum_admitted_families": 4}
    )
    assert not audit._quota_contract(
        {"minimum_admitted_scenes": 5, "minimum_admitted_families": 4}
    )
    assert not audit._quota_contract(
        {"minimum_admitted_scenes": 6, "minimum_admitted_families": 3}
    )
