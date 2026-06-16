"""CenterNet losses in JAX: penalty-reduced focal loss + masked L1 regression."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def focal_loss(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    alpha: float = 2.0,
    beta: float = 4.0,
    sample_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Penalty-reduced pixel-wise focal loss (CornerNet/CenterNet).

    Computed from logits with ``log_sigmoid`` for numerical stability.
    """
    pred = jax.nn.sigmoid(logits)
    log_p = jax.nn.log_sigmoid(logits)
    log_1mp = jax.nn.log_sigmoid(-logits)
    pos = (target >= 1.0 - 1e-4).astype(jnp.float32)
    neg = 1.0 - pos
    pos_term = log_p * jnp.power(1.0 - pred, alpha) * pos
    neg_term = log_1mp * jnp.power(pred, alpha) * jnp.power(1.0 - target, beta) * neg
    if sample_weight is not None:
        w = sample_weight.reshape((-1,) + (1,) * (logits.ndim - 1)).astype(jnp.float32)
        pos, pos_term, neg_term = pos * w, pos_term * w, neg_term * w
    num_pos = jnp.maximum(pos.sum(), 1.0)
    return -(pos_term.sum() + neg_term.sum()) / num_pos


def gather_feature(feature_map: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """(B, H, W, D) gathered at flat (B, M) spatial indices -> (B, M, D)."""
    b, h, w, d = feature_map.shape
    flat = feature_map.reshape(b, h * w, d)
    return jnp.take_along_axis(flat, indices[..., None].astype(jnp.int32), axis=1)


def masked_l1(pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    m = mask[..., None]
    return jnp.sum(jnp.abs(pred - target) * m) / (jnp.sum(m) * pred.shape[-1] + 1e-4)


def centernet_loss(outputs: dict, batch: dict, loss_cfg) -> tuple[jnp.ndarray, dict]:
    valid = batch.get("valid")
    weight = None if valid is None else valid.astype(jnp.float32)
    hm = focal_loss(outputs["heatmap"], batch["heatmap"], loss_cfg.focal_alpha, loss_cfg.focal_beta, weight)
    mask = batch["reg_mask"]
    if weight is not None:
        mask = mask * weight[:, None]
    size = masked_l1(gather_feature(outputs["size"], batch["indices"]), batch["size"], mask)
    off = masked_l1(gather_feature(outputs["offset"], batch["indices"]), batch["offset"], mask)
    total = (
        loss_cfg.heatmap_weight * hm + loss_cfg.size_weight * size + loss_cfg.offset_weight * off
    )
    return total, {"loss": total, "heatmap_loss": hm, "size_loss": size, "offset_loss": off}
