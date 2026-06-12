"""Non-maximum suppression: NumPy reference and a jittable fixed-size JAX version.

The C++ implementation lives in ``cpp/src/nms.cpp`` and is dispatched through
:func:`perceptnet.ops.cpp_ext.nms`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from perceptnet.ops.boxes import iou_matrix_jax


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS via the coordinate-offset trick (boxes of different classes never overlap)."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    max_coord = float(boxes.max()) + 1.0
    offsets = labels.astype(np.float32)[:, None] * max_coord
    return nms_numpy(boxes + offsets, scores, iou_threshold)


@partial(jax.jit, static_argnames=("iou_threshold",))
def nms_jax(boxes: jnp.ndarray, scores: jnp.ndarray, iou_threshold: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-size greedy NMS for use inside compiled graphs.

    Args:
        boxes: (K, 4) boxes.
        scores: (K,) scores. Entries with score <= 0 are treated as padding.

    Returns:
        ``(order, keep)`` where ``order`` sorts the inputs by descending score and ``keep`` is a
        boolean mask over the sorted boxes. Output shapes are static, so it lowers to a single
        XLA ``while`` loop.
    """
    order = jnp.argsort(-scores)
    b = boxes[order]
    s = scores[order]
    k = b.shape[0]
    iou = iou_matrix_jax(b, b)
    idx = jnp.arange(k)

    def body(i, keep):
        suppressed = jnp.any((iou[:, i] > iou_threshold) & keep & (idx < i))
        return keep.at[i].set(keep[i] & ~suppressed)

    keep = jax.lax.fori_loop(0, k, body, s > 0)
    return order, keep
