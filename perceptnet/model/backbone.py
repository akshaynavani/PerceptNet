"""ResNet backbones in Flax (NHWC), returning multi-scale feature maps.

Module/parameter names deliberately mirror torchvision's ResNet layout (``conv1``, ``bn1``,
``layer1_0`` ...) so ImageNet weights can be converted with a simple key mapping
(see ``scripts/convert_torchvision_resnet.py``).
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

ModuleDef = Any

_STAGES = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}
_WIDTHS = (64, 128, 256, 512)


class BasicBlock(nn.Module):
    filters: int
    strides: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        norm = lambda name: nn.BatchNorm(  # noqa: E731
            use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype, name=name
        )
        residual = x
        y = nn.Conv(self.filters, (3, 3), (self.strides, self.strides), padding=((1, 1), (1, 1)),
                    use_bias=False, dtype=self.dtype, name="conv1")(x)
        y = nn.relu(norm("bn1")(y))
        y = nn.Conv(self.filters, (3, 3), padding=((1, 1), (1, 1)), use_bias=False,
                    dtype=self.dtype, name="conv2")(y)
        y = norm("bn2")(y)
        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), (self.strides, self.strides), use_bias=False,
                               dtype=self.dtype, name="downsample_conv")(residual)
            residual = norm("downsample_bn")(residual)
        return nn.relu(y + residual)


class ResNet(nn.Module):
    """Returns ``[C2, C3, C4, C5]`` at strides 4, 8, 16, 32."""

    stage_sizes: tuple[int, ...]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(64, (7, 7), (2, 2), padding=((3, 3), (3, 3)), use_bias=False,
                    dtype=self.dtype, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5,
                         dtype=self.dtype, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        features = []
        for stage, (n_blocks, width) in enumerate(zip(self.stage_sizes, _WIDTHS)):
            for b in range(n_blocks):
                strides = 2 if stage > 0 and b == 0 else 1
                x = BasicBlock(width, strides, dtype=self.dtype, name=f"layer{stage + 1}_{b}")(x, train)
            features.append(x)
        return features


def build_backbone(name: str, dtype=jnp.float32) -> ResNet:
    if name not in _STAGES:
        raise ValueError(f"Unknown backbone '{name}', choose from {sorted(_STAGES)}")
    return ResNet(stage_sizes=_STAGES[name], dtype=dtype)
