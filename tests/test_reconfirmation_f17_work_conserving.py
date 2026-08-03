from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts import run_kinofail_reconfirmation_scene_pipeline_v3 as pipeline
from scripts.kinofail_reconfirmation_slot_pool_v1 import (
    _FairSlotPool,
    connect_slot_pool,
    start_slot_broker,
)


def _slot_worker(
    environment: dict[str, str],
    queue: mp.Queue,
    worker_id: int,
) -> None:
    os.environ.update(environment)
    manager, slot_pool = connect_slot_pool()
    ticket = int(slot_pool.acquire())
    queue.put(("start", worker_id, ticket, time.monotonic()))
    time.sleep(0.1)
    queue.put(("end", worker_id, ticket, time.monotonic()))
    slot_pool.release(ticket)
    _ = manager


def test_f17_slot_pool_never_exceeds_three() -> None:
    broker = start_slot_broker(3)
    queue: mp.Queue = mp.Queue()
    workers = [
        mp.Process(
            target=_slot_worker,
            args=(broker.environment, queue, index),
        )
        for index in range(7)
    ]
    try:
        for worker in workers:
            worker.start()
        events = [queue.get(timeout=5) for _ in range(2 * len(workers))]
        for worker in workers:
            worker.join(timeout=5)
    finally:
        broker.shutdown()

    active = 0
    peak = 0
    tickets = set()
    for event, _, ticket, timestamp in sorted(events, key=lambda row: row[3]):
        active += 1 if event == "start" else -1
        peak = max(peak, active)
        if event == "start":
            tickets.add(ticket)
    assert peak == 3
    assert active == 0
    assert tickets == set(range(len(workers)))
    assert all(worker.exitcode == 0 for worker in workers)


def test_f17_slot_pool_admits_waiters_fifo() -> None:
    pool = _FairSlotPool(1)
    blocker = pool.acquire()
    admitted: list[int] = []

    def contender() -> None:
        ticket = pool.acquire()
        admitted.append(ticket)
        pool.release(ticket)

    workers = []
    for expected_waiters in range(1, 5):
        worker = threading.Thread(target=contender)
        worker.start()
        workers.append(worker)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with pool._condition:
                if len(pool._waiting) == expected_waiters:
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("contender did not enter the FIFO queue")

    pool.release(blocker)
    for worker in workers:
        worker.join(timeout=2)
    assert admitted == [1, 2, 3, 4]
    assert all(not worker.is_alive() for worker in workers)


def test_f17_preserves_four_original_logical_partitions() -> None:
    scene = "confirm_v2_production_scene_04"
    paths = pipeline.base._scene_paths(scene)
    commands = [
        pipeline._pair_command(
            paths=paths,
            battery="scale",
            partition_index=index,
        )
        for index in range(4)
    ]
    assert pipeline.LOGICAL_PARTITION_COUNT == 4
    assert pipeline.MAX_CONCURRENT_ISAAC_PROCESSES == 3
    assert {
        command[command.index("--partition-index") + 1]
        for command in commands
    } == {"0", "1", "2", "3"}
    assert all(
        command[command.index("--partition-count") + 1] == "4"
        for command in commands
    )
    assert all(
        Path(command[1]).name
        == "run_kinofail_reconfirmation_slotted_partition_v1.py"
        for command in commands
    )


def test_f17_reuses_exact_existing_scientific_runners() -> None:
    partition_source = pipeline.SLOTTED_PARTITION.read_text(encoding="utf-8")
    t2_source = pipeline.SLOTTED_T2.read_text(encoding="utf-8")
    assert (
        "run_kinofail_reconfirmation_pair_partition_v2 as base"
        in partition_source
    )
    assert "run_kinofail_reconfirmation_t2_scene_v2 as base" in t2_source
    assert "base.subprocess.run = slotted_run" in partition_source
    assert "base.subprocess.run = slotted_run" in t2_source


def test_f17_production_entrypoints_resolve_repo_imports() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    scene = subprocess.run(
        [sys.executable, str(Path(pipeline.__file__)), "--help"],
        cwd=pipeline.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert scene.returncode == 0, scene.stderr

    for script in (pipeline.SLOTTED_PARTITION, pipeline.SLOTTED_T2):
        guarded = subprocess.run(
            [sys.executable, str(script)],
            cwd=pipeline.ROOT,
            env={
                key: value
                for key, value in environment.items()
                if not key.startswith("KINOVLA_RECONFIRMATION_SLOT_")
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert "slot broker environment is incomplete" in guarded.stderr
        assert "ImportError" not in guarded.stderr


def test_f17_slotted_entrypoints_connect_to_live_broker() -> None:
    broker = start_slot_broker(3)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(broker.environment)
    try:
        for script in (pipeline.SLOTTED_PARTITION, pipeline.SLOTTED_T2):
            connected = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=pipeline.ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert connected.returncode == 0, connected.stderr
            assert "usage:" in connected.stdout
    finally:
        broker.shutdown()
