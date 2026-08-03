#!/usr/bin/env python3
"""Process-shared three-slot broker for reconfirmation Isaac launches.

The broker limits *Isaac child processes*, not long-lived logical partition
runners.  This lets all four frozen partitions remain eligible to make
progress while preserving the F7 maximum of three concurrent Isaac processes.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from typing import Any


HOST_ENV = "KINOVLA_RECONFIRMATION_SLOT_HOST"
PORT_ENV = "KINOVLA_RECONFIRMATION_SLOT_PORT"
AUTHKEY_ENV = "KINOVLA_RECONFIRMATION_SLOT_AUTHKEY_HEX"
CAPACITY_ENV = "KINOVLA_RECONFIRMATION_SLOT_CAPACITY"

class _FairSlotPool:
    """FIFO admission to a fixed number of process slots."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._waiting: deque[int] = deque()
        self._active: set[int] = set()

    def acquire(self) -> int:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting.append(ticket)
            while (
                len(self._active) >= self._capacity
                or self._waiting[0] != ticket
            ):
                self._condition.wait()
            self._waiting.popleft()
            self._active.add(ticket)
            self._condition.notify_all()
            return ticket

    def release(self, ticket: int) -> None:
        with self._condition:
            if ticket not in self._active:
                raise RuntimeError(f"slot ticket is not active: {ticket}")
            self._active.remove(ticket)
            self._condition.notify_all()


_SERVER_SLOT_POOL: _FairSlotPool | None = None


def _server_slot_pool() -> _FairSlotPool:
    if _SERVER_SLOT_POOL is None:
        raise RuntimeError("slot broker pool is not initialized")
    return _SERVER_SLOT_POOL


class _ServerManager(BaseManager):
    pass


class _ClientManager(BaseManager):
    pass


@dataclass
class SlotBroker:
    manager: BaseManager
    environment: dict[str, str]
    capacity: int

    def shutdown(self) -> None:
        self.manager.shutdown()


def start_slot_broker(capacity: int = 3) -> SlotBroker:
    """Start one local broker and return environment for child launchers."""

    if capacity != 3:
        raise ValueError("the F7 stable-capacity contract requires three slots")
    global _SERVER_SLOT_POOL
    _SERVER_SLOT_POOL = _FairSlotPool(capacity)
    _ServerManager.register(
        "get_slot_pool",
        callable=_server_slot_pool,
        exposed=("acquire", "release"),
    )
    authkey = os.urandom(32)
    manager = _ServerManager(address=("127.0.0.1", 0), authkey=authkey)
    manager.start()
    host, port = manager.address
    environment = {
        HOST_ENV: str(host),
        PORT_ENV: str(port),
        AUTHKEY_ENV: authkey.hex(),
        CAPACITY_ENV: str(capacity),
    }
    return SlotBroker(
        manager=manager,
        environment=environment,
        capacity=capacity,
    )


def connect_slot_pool() -> Any:
    """Connect a child launcher to the broker described by its environment."""

    missing = [
        key
        for key in (HOST_ENV, PORT_ENV, AUTHKEY_ENV, CAPACITY_ENV)
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(f"slot broker environment is incomplete: {missing}")
    if int(os.environ[CAPACITY_ENV]) != 3:
        raise RuntimeError("slot broker capacity differs from the F7 contract")
    _ClientManager.register("get_slot_pool")
    manager = _ClientManager(
        address=(os.environ[HOST_ENV], int(os.environ[PORT_ENV])),
        authkey=bytes.fromhex(os.environ[AUTHKEY_ENV]),
    )
    manager.connect()
    return manager, manager.get_slot_pool()
