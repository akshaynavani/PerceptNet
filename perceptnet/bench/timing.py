"""Timing helpers that account for JAX's asynchronous dispatch."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np


def block(x):
    """Wait for every array in a pytree (JAX dispatch is async; timing without this lies)."""
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


@dataclass
class TimingResult:
    name: str
    shape: str
    backend: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    iters: int

    def row(self) -> dict:
        return self.__dict__.copy()


def time_fn(fn: Callable[[], object], name: str, shape: str, backend: str,
            iters: int = 50, warmup: int = 3) -> TimingResult:
    for _ in range(warmup):
        block(fn())
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        block(fn())
        times[i] = (time.perf_counter() - t0) * 1e3
    return TimingResult(name, shape, backend, float(np.percentile(times, 50)),
                        float(np.percentile(times, 95)), float(times.mean()), iters)


def format_table(results: list[TimingResult]) -> str:
    header = f"{'op':<22}{'shape':<26}{'backend':<14}{'p50 ms':>10}{'p95 ms':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r.name:<22}{r.shape:<26}{r.backend:<14}{r.p50_ms:>10.3f}{r.p95_ms:>10.3f}")
    return "\n".join(lines)
