"""Optimizer construction with backbone-specific fine-tuning controls.

* Linear warmup + cosine decay schedule.
* Global-norm gradient clipping.
* ``optax.multi_transform`` splits params into ``backbone`` and ``head`` groups: the backbone
  gets a reduced learning rate (``backbone_lr_mult``) and receives exactly zero updates for the
  first ``freeze_backbone_steps`` steps, so freshly initialized heads can settle before the
  pretrained features are disturbed.
* Weight decay is applied only to conv/dense kernels (not biases or BatchNorm scales).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util


def make_schedule(base_lr: float, warmup_steps: int, total_steps: int) -> optax.Schedule:
    warmup_steps = max(1, min(warmup_steps, total_steps - 1)) if total_steps > 1 else 1
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=base_lr * 0.01,
    )


def param_groups(params) -> dict:
    flat = traverse_util.flatten_dict(params)
    labels = {k: ("backbone" if k[0] == "backbone" else "head") for k in flat}
    return traverse_util.unflatten_dict(labels)


def decay_mask(params) -> dict:
    flat = traverse_util.flatten_dict(params)
    return traverse_util.unflatten_dict({k: (k[-1] == "kernel") for k in flat})


def build_optimizer(train_cfg, total_steps: int) -> tuple[optax.GradientTransformation, optax.Schedule]:
    head_schedule = make_schedule(train_cfg.lr, train_cfg.warmup_steps, total_steps)
    freeze = int(train_cfg.freeze_backbone_steps)
    mult = float(train_cfg.backbone_lr_mult)

    def backbone_schedule(step):
        lr = head_schedule(step) * mult
        return jnp.where(step < freeze, 0.0, lr)

    def adamw(schedule):
        return optax.adamw(schedule, b1=0.9, b2=0.999, weight_decay=train_cfg.weight_decay, mask=decay_mask)

    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.multi_transform(
            {"backbone": adamw(backbone_schedule), "head": adamw(head_schedule)},
            param_groups,
        ),
    )
    return tx, head_schedule


def frozen_fraction(params) -> float:
    """Fraction of parameters that belong to the backbone (for logging)."""
    flat = traverse_util.flatten_dict(params)
    total = sum(v.size for v in flat.values())
    bb = sum(v.size for k, v in flat.items() if k[0] == "backbone")
    return bb / max(total, 1)


__all__ = ["build_optimizer", "make_schedule", "param_groups", "decay_mask", "frozen_fraction", "jax"]
