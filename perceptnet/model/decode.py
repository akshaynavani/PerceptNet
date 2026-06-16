"""Jittable CenterNet decoding: peak extraction + top-K + box assembly, all in XLA."""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from perceptnet.model.losses import gather_feature


@partial(jax.jit, static_argnames=("top_k", "stride", "pool_size"))
def decode(outputs: dict, top_k: int = 100, stride: int = 4, pool_size: int = 3):
    """Convert raw head outputs to detections in network-input pixel coordinates.

    A 3x3 max-pool acts as a cheap NMS on the heatmap: only local maxima survive.

    Returns:
        boxes (B, K, 4) xyxy, scores (B, K), classes (B, K) int32.
    """
    heat = jax.nn.sigmoid(outputs["heatmap"])
    b, h, w, c = heat.shape
    hmax = nn.max_pool(heat, (pool_size, pool_size), strides=(1, 1), padding="SAME")
    heat = jnp.where(heat == hmax, heat, 0.0)

    k = min(top_k, h * w * c)
    scores, flat_idx = jax.lax.top_k(heat.reshape(b, -1), k)
    classes = (flat_idx % c).astype(jnp.int32)
    pix = flat_idx // c
    ys = (pix // w).astype(jnp.float32)
    xs = (pix % w).astype(jnp.float32)

    offset = gather_feature(outputs["offset"], pix)
    size = gather_feature(outputs["size"], pix)
    cx = (xs + offset[..., 0]) * stride
    cy = (ys + offset[..., 1]) * stride
    bw = jnp.maximum(size[..., 0], 0.0) * stride
    bh = jnp.maximum(size[..., 1], 0.0) * stride
    boxes = jnp.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1)
    return boxes, scores, classes
