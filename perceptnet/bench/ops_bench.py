"""Numeric-library micro-benchmarks: NumPy vs JAX (XLA-jitted) vs C++.

Covers the operations that dominate a perception pipeline outside the backbone:
box IoU, NMS, frame preprocessing, decode-style elementwise/pooling, and dense matmul.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from perceptnet.bench.timing import TimingResult, time_fn
from perceptnet.data.preprocess import preprocess_frame
from perceptnet.ops import cpp_ext
from perceptnet.ops.boxes import iou_matrix, iou_matrix_jax
from perceptnet.ops.nms import nms_jax, nms_numpy

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def clustered_boxes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random boxes grouped in clusters of ~20 so NMS actually suppresses things."""
    rng = np.random.default_rng(seed)
    k = max(1, n // 20)
    centers = rng.uniform(0, 1000, (k, 2))
    sizes = rng.uniform(20, 120, (k, 2))
    c = np.arange(n) % k
    xy = centers[c] + rng.normal(0, 8, (n, 2))
    wh = sizes[c]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    labels = rng.integers(0, 3, n).astype(np.int32)
    return boxes, scores, labels


def bench_iou(sizes=(100, 1000), iters=50) -> list[TimingResult]:
    out = []
    iou_jit = jax.jit(iou_matrix_jax)
    for n in sizes:
        b, _, _ = clustered_boxes(n)
        bj = jnp.asarray(b)
        shape = f"{n}x{n}"
        out.append(time_fn(lambda: iou_matrix(b, b), "iou_matrix", shape, "numpy", iters))
        out.append(time_fn(lambda: iou_jit(bj, bj), "iou_matrix", shape, f"jax-{jax.default_backend()}", iters))
        if cpp_ext.HAS_CPP:
            out.append(time_fn(lambda: cpp_ext.iou_matrix(b, b, backend="cpp"), "iou_matrix", shape, "cpp", iters))
    return out


def bench_nms(sizes=(100, 1000, 3000), iters=20) -> list[TimingResult]:
    out = []
    for n in sizes:
        b, s, _ = clustered_boxes(n)
        bj, sj = jnp.asarray(b), jnp.asarray(s)
        shape = f"N={n}"
        out.append(time_fn(lambda: nms_numpy(b, s, 0.5), "nms", shape, "numpy", iters))
        out.append(time_fn(lambda: nms_jax(bj, sj, 0.5), "nms", shape, f"jax-{jax.default_backend()}", iters))
        if cpp_ext.HAS_CPP:
            out.append(time_fn(lambda: cpp_ext.nms(b, s, 0.5, backend="cpp"), "nms", shape, "cpp", iters))
    return out


def bench_preprocess(targets=((384, 640), (384, 1248)), iters=30) -> list[TimingResult]:
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    out = []
    for hw in targets:
        shape = f"1080p->{hw[0]}x{hw[1]}"
        out.append(time_fn(lambda: preprocess_frame(frame, hw, MEAN, STD), "preprocess", shape, "opencv+numpy", iters))
        if cpp_ext.HAS_CPP_OPENCV:
            out.append(time_fn(lambda: cpp_ext.preprocess(frame, hw, MEAN, STD, backend="cpp"),
                               "preprocess", shape, "cpp", iters))
    return out


def bench_decode_ops(hw=(96, 160), classes=3, iters=30) -> list[TimingResult]:
    """Sigmoid + 3x3 max-pool peak mask + top-k over a heatmap (the CenterNet decode core)."""
    rng = np.random.default_rng(0)
    heat = rng.normal(size=(1, *hw, classes)).astype(np.float32)

    def numpy_decode(x):
        p = 1.0 / (1.0 + np.exp(-x))
        padded = np.pad(p, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        peaks = np.where(p == windows.max(axis=(-1, -2)), p, 0.0).reshape(1, -1)
        idx = np.argpartition(-peaks, 100, axis=1)[:, :100]
        return np.take_along_axis(peaks, idx, axis=1)

    @jax.jit
    def jax_decode(x):
        p = jax.nn.sigmoid(x)
        m = jax.lax.reduce_window(p, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "SAME")
        peaks = jnp.where(p == m, p, 0.0).reshape(1, -1)
        return jax.lax.top_k(peaks, 100)[0]

    hj = jnp.asarray(heat)
    shape = f"{hw[0]}x{hw[1]}x{classes}"
    return [
        time_fn(lambda: numpy_decode(heat), "heatmap_decode", shape, "numpy", iters),
        time_fn(lambda: jax_decode(hj), "heatmap_decode", shape, f"jax-{jax.default_backend()}", iters),
    ]


def bench_matmul(sizes=(512, 1024, 2048), iters=20) -> list[TimingResult]:
    out = []
    mm = jax.jit(jnp.matmul)
    rng = np.random.default_rng(0)
    for n in sizes:
        a = rng.normal(size=(n, n)).astype(np.float32)
        aj = jnp.asarray(a)
        shape = f"{n}x{n} @ {n}x{n}"
        out.append(time_fn(lambda: a @ a, "matmul", shape, "numpy", iters))
        out.append(time_fn(lambda: mm(aj, aj), "matmul", shape, f"jax-{jax.default_backend()}", iters))
    return out


def run_all(quick: bool = False) -> list[TimingResult]:
    it = 5 if quick else None
    kw = (lambda d: {"iters": it} if it else {})
    results = []
    results += bench_iou(**kw(None))
    results += bench_nms(sizes=(100, 1000) if quick else (100, 1000, 3000), **kw(None))
    results += bench_preprocess(**kw(None))
    results += bench_decode_ops(**kw(None))
    results += bench_matmul(sizes=(512,) if quick else (512, 1024, 2048), **kw(None))
    return results
