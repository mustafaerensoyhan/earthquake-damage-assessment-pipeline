"""
Timing utilities for latency benchmarking.

Provides a context-manager timer and functions for computing
median / P95 statistics across repeated measurements.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List

import numpy as np


@contextmanager
def timer():
    """
    Context manager that yields a dict and fills in ``elapsed_ms``
    on exit.

    Usage::

        with timer() as t:
            result = model(x)
        print(t["elapsed_ms"])
    """
    record = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    yield record
    record["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def latency_stats(times_ms: List[float]) -> dict:
    """
    Compute summary statistics for a list of latency measurements.

    Returns dict with keys: median, mean, p95, p99, min, max, std, count.
    """
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
        "count": len(arr),
    }
