from __future__ import annotations

from scripts.recover_kinofail_reconfirmation_scene01_partitions_f15 import (
    partition_pair_ids,
)


def test_f15_partitions_are_disjoint_and_cover_scale_schedule() -> None:
    partitions = [partition_pair_ids(index) for index in range(4)]

    assert all(len(values) == 88 for values in partitions)
    flattened = [pair_id for values in partitions for pair_id in values]
    assert len(flattened) == 352
    assert len(set(flattened)) == 352
