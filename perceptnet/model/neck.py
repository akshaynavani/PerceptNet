"""Upsampling neck that fuses ResNet features into a single stride-4 map (CenterNet/FPN hybrid)."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def upsample2x(x: jnp.ndarray) -> jnp.ndarray:
    b, h, w, c = x.shape
    return jax.image.resize(x, (b, h * 2, w * 2, c), method="bilinear")


class UpsampleNeck(nn.Module):
    channels: tuple[int, ...] = (256, 128, 64)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features: list[jnp.ndarray], train: bool) -> jnp.ndarray:
        # features: [C2, C3, C4, C5]; walk top-down C5 -> C4 -> C3 -> C2.
        laterals = features[:-1][::-1]  # C4, C3, C2
        x = features[-1]
        if len(self.channels) != len(laterals):
            raise ValueError("neck_channels must have one entry per upsampling stage (3)")
        for i, (ch, lateral) in enumerate(zip(self.channels, laterals)):
            x = nn.Conv(ch, (3, 3), padding="SAME", use_bias=False, dtype=self.dtype, name=f"up{i}_conv")(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype, name=f"up{i}_bn")(x)
            x = nn.relu(x)
            x = upsample2x(x)
            lat = nn.Conv(ch, (1, 1), use_bias=False, dtype=self.dtype, name=f"lat{i}_conv")(lateral)
            lat = nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype, name=f"lat{i}_bn")(lat)
            x = nn.relu(x + lat)
        return x
