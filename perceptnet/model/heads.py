"""CenterNet prediction heads: class heatmap, box size and sub-pixel center offset."""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class Head(nn.Module):
    out_channels: int
    hidden: int = 64
    bias_init_value: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden, (3, 3), padding="SAME", dtype=self.dtype, name="conv")(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.out_channels,
            (1, 1),
            dtype=self.dtype,
            bias_init=nn.initializers.constant(self.bias_init_value),
            kernel_init=nn.initializers.normal(0.001),
            name="out",
        )(x)
        return x.astype(jnp.float32)


class CenterNetHeads(nn.Module):
    num_classes: int
    hidden: int = 64
    heatmap_prior: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        prior_bias = -math.log((1.0 - self.heatmap_prior) / self.heatmap_prior)
        return {
            "heatmap": Head(self.num_classes, self.hidden, prior_bias, self.dtype, name="heatmap")(x),
            "size": Head(2, self.hidden, dtype=self.dtype, name="size")(x),
            "offset": Head(2, self.hidden, dtype=self.dtype, name="offset")(x),
        }
