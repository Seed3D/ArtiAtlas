# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Lightweight per-sample phase timing.

Usage at the orchestrator level (one timer per sample):

    from artibench.utils.timing import PhaseTimer, push_timer, pop_timer

    timer = PhaseTimer(tag=obj_id)
    push_timer(timer)
    try:
        ... do work ...
    finally:
        pop_timer()
        print(timer.summary())

Inside any function along the call chain, record sub-phases via the
`phase(name)` context manager; it is a no-op when no timer is active.

    from artibench.utils.timing import phase
    with phase("align.clip"):
        compute_clip_score(...)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

__all__ = [
    "PhaseTimer",
    "phase",
    "push_timer",
    "pop_timer",
    "current_timer",
]


class PhaseTimer:
    """Accumulates (phase_name, elapsed_sec) and renders a one-line summary."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self.entries: List[Tuple[str, float]] = []
        self._start = time.perf_counter()

    def record(self, name: str, seconds: float) -> None:
        self.entries.append((name, float(seconds)))

    def total(self) -> float:
        return time.perf_counter() - self._start

    def summary(self) -> str:
        parts = [f"{n}={s:.2f}s" for n, s in self.entries]
        return (
            f"[timing:{self.tag}] "
            + " ".join(parts)
            + f" | total={self.total():.2f}s"
        )


_STACK: List[PhaseTimer] = []


def push_timer(timer: PhaseTimer) -> None:
    _STACK.append(timer)


def pop_timer() -> Optional[PhaseTimer]:
    return _STACK.pop() if _STACK else None


def current_timer() -> Optional[PhaseTimer]:
    return _STACK[-1] if _STACK else None


@contextmanager
def phase(name: str):
    """Record a phase on the active timer; silent no-op if no timer is active."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        timer = _STACK[-1] if _STACK else None
        if timer is not None:
            timer.record(name, dt)
