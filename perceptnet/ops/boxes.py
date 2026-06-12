"""Box utilities with matching NumPy and JAX implementations.

Boxes are ``xyxy`` everywhere. The NumPy versions are the reference implementation used for
evaluation and parity tests; the JAX versions are jittable and used inside compiled graphs and
in the numeric-library benchmarks.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[..., 2] - boxes[..., 0], 0, None) * np.clip(
        boxes[..., 3] - boxes[..., 1], 0, None
    )


def iou_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Pairwise IoU between ``a`` (N,4) and ``b`` (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / np.maximum(union, eps)


def iou_matrix_jax(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    lt = jnp.maximum(a[:, None, :2], b[None, :, :2])
    rb = jnp.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = jnp.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = jnp.clip(a[:, 2] - a[:, 0], 0) * jnp.clip(a[:, 3] - a[:, 1], 0)
    area_b = jnp.clip(b[:, 2] - b[:, 0], 0) * jnp.clip(b[:, 3] - b[:, 1], 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / jnp.maximum(union, eps)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = (boxes[..., i] for i in range(4))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
