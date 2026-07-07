"""End-to-end model latency: eager op-by-op vs jax.jit vs AOT-compiled XLA, across batch sizes."""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from perceptnet.bench.timing import TimingResult, block, time_fn
from perceptnet.config import Config
from perceptnet.model.decode import decode
from perceptnet.model.detector import build_model, init_variables

log = logging.getLogger(__name__)


def bench_latency(
    cfg: Config,
    variables: dict | None = None,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    iters: int = 30,
    include_eager: bool = True,
) -> tuple[list[TimingResult], list[dict]]:
    """Returns timing rows and a per-batch-size summary (compile time, throughput)."""
    model = build_model(cfg)
    hw = tuple(cfg.data.input_size)
    if variables is None:
        variables = init_variables(model, hw, jax.random.PRNGKey(0))
    variables = jax.device_put(variables)
    top_k, stride = cfg.inference.top_k, cfg.model.output_stride

    def forward(v, x):
        return decode(model.apply(v, x, train=False), top_k=top_k, stride=stride)

    backend = jax.default_backend()
    rows: list[TimingResult] = []
    summary: list[dict] = []
    for b in batch_sizes:
        x = jnp.asarray(np.random.default_rng(0).normal(size=(b, *hw, 3)).astype(np.float32))
        shape = f"{b}x{hw[0]}x{hw[1]}"

        jitted = jax.jit(forward)
        t0 = time.perf_counter()
        block(jitted(variables, x))  # first call = trace + compile + run
        jit_first_call = time.perf_counter() - t0
        rows.append(time_fn(lambda: jitted(variables, x), "e2e_forward", shape, f"jit-{backend}", iters))

        t0 = time.perf_counter()
        compiled = jax.jit(forward).lower(variables, x).compile()
        aot_compile = time.perf_counter() - t0
        aot = time_fn(lambda: compiled(variables, x), "e2e_forward", shape, f"aot-{backend}", iters)
        rows.append(aot)

        if include_eager:
            with jax.disable_jit():
                rows.append(time_fn(lambda: forward(variables, x), "e2e_forward", shape,
                                    f"eager-{backend}", max(3, iters // 10), warmup=1))

        summary.append({
            "batch_size": b,
            "jit_first_call_s": jit_first_call,
            "aot_compile_s": aot_compile,
            "aot_p50_ms": aot.p50_ms,
            "aot_images_per_s": b / (aot.p50_ms / 1e3),
        })
        log.info("batch %d: AOT p50 %.2f ms (%.1f img/s), compile %.2fs", b, aot.p50_ms,
                 summary[-1]["aot_images_per_s"], aot_compile)
    return rows, summary
