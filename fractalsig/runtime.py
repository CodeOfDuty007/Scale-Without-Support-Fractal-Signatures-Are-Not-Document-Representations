"""Runtime knobs shared by the experiment drivers."""
from __future__ import annotations

import os


def default_workers() -> int:
    """Worker-process count: $FRACTALSIG_WORKERS if set, else all visible CPUs.

    Results do not depend on this value: every stochastic task is seeded per
    task, not per worker.
    """
    raw = os.environ.get("FRACTALSIG_WORKERS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError as exc:
            raise ValueError(f"FRACTALSIG_WORKERS must be an integer, got {raw!r}") from exc
        if n < 1:
            raise ValueError(f"FRACTALSIG_WORKERS must be >= 1, got {n}")
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS / Windows
        return max(1, os.cpu_count() or 1)
