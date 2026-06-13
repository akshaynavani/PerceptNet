"""The PerceptNet detector: ResNet backbone -> upsampling neck -> CenterNet heads."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from perceptnet.model.backbone import build_backbone
from perceptnet.model.heads import CenterNetHeads
from perceptnet.model.neck import UpsampleNeck

_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


class PerceptNet(nn.Module):
    num_classes: int
    backbone: str = "resnet18"
    neck_channels: tuple[int, ...] = (256, 128, 64)
    head_channels: int = 64
    heatmap_prior: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images: jnp.ndarray, train: bool = False, backbone_bn_train: bool | None = None):
        """
        Args:
            images: (B, H, W, 3) normalized float input; H and W must be divisible by 32.
            train: enables BatchNorm batch statistics in neck/heads (and backbone by default).
            backbone_bn_train: overrides BN mode for the backbone only (freeze BN while fine-tuning).

        Returns:
            dict with ``heatmap`` (B, H/4, W/4, C) logits, ``size`` (B, H/4, W/4, 2) and
            ``offset`` (B, H/4, W/4, 2), all float32.
        """
        bb_train = train if backbone_bn_train is None else backbone_bn_train
        x = images.astype(self.dtype)
        feats = build_backbone(self.backbone, self.dtype).clone(name="backbone")(x, bb_train)
        fused = UpsampleNeck(self.neck_channels, self.dtype, name="neck")(feats, train)
        return CenterNetHeads(
            self.num_classes, self.head_channels, self.heatmap_prior, self.dtype, name="heads"
        )(fused)


def build_model(cfg) -> PerceptNet:
    """Build from a full :class:`~perceptnet.config.Config`."""
    if cfg.model.output_stride != 4:
        raise ValueError("PerceptNet's neck produces stride-4 outputs; output_stride must be 4")
    h, w = cfg.data.input_size
    if h % 32 or w % 32:
        raise ValueError(f"input_size {cfg.data.input_size} must be divisible by 32")
    return PerceptNet(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.backbone,
        neck_channels=tuple(cfg.model.neck_channels),
        head_channels=cfg.model.head_channels,
        heatmap_prior=cfg.model.heatmap_prior,
        dtype=_DTYPES[cfg.model.dtype],
    )


def init_variables(model: PerceptNet, input_hw: tuple[int, int], rng: jax.Array, batch: int = 1):
    dummy = jnp.zeros((batch, *input_hw, 3), jnp.float32)
    return model.init(rng, dummy, train=False)


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
