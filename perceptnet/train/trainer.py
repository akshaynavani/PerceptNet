"""Training loop: jitted train/eval steps, logging, checkpointing and periodic evaluation."""

from __future__ import annotations

import json
import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from perceptnet.config import Config
from perceptnet.data.dataset import build_dataset
from perceptnet.data.loader import DataLoader
from perceptnet.model.detector import build_model, count_params, init_variables
from perceptnet.model.losses import centernet_loss
from perceptnet.model.targets import CenterNetTargetEncoder
from perceptnet.train.checkpoint import save_checkpoint, save_inference_variables
from perceptnet.train.optim import build_optimizer, frozen_fraction
from perceptnet.train.pretrained import load_pretrained_backbone
from perceptnet.train.state import TrainState, create_train_state

log = logging.getLogger(__name__)

TARGET_KEYS = ("image", "heatmap", "indices", "size", "offset", "reg_mask", "valid")


def device_batch(batch: dict) -> dict:
    return {k: jnp.asarray(batch[k]) for k in TARGET_KEYS if k in batch}


@partial(jax.jit, static_argnames=("loss_cfg", "backbone_bn_train"), donate_argnums=(0,))
def train_step(state: TrainState, batch: dict, loss_cfg, backbone_bn_train: bool | None):
    def loss_fn(params):
        outputs, mutated = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch["image"],
            train=True,
            backbone_bn_train=backbone_bn_train,
            mutable=["batch_stats"],
        )
        loss, metrics = centernet_loss(outputs, batch, loss_cfg)
        return loss, (metrics, mutated["batch_stats"])

    (_, (metrics, new_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics["grad_norm"] = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree_util.tree_leaves(grads)))
    state = state.apply_gradients(grads=grads, batch_stats=new_stats)
    return state, metrics


@partial(jax.jit, static_argnames=("loss_cfg",))
def eval_step(state: TrainState, batch: dict, loss_cfg):
    outputs = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch["image"], train=False
    )
    _, metrics = centernet_loss(outputs, batch, loss_cfg)
    return outputs, metrics


class _HashableLossCfg:
    """Wrap the loss dataclass so it can be a static jit argument (dataclasses aren't hashable)."""

    def __init__(self, cfg) -> None:
        self.heatmap_weight = float(cfg.heatmap_weight)
        self.size_weight = float(cfg.size_weight)
        self.offset_weight = float(cfg.offset_weight)
        self.focal_alpha = float(cfg.focal_alpha)
        self.focal_beta = float(cfg.focal_beta)
        self._key = (self.heatmap_weight, self.size_weight, self.offset_weight, self.focal_alpha, self.focal_beta)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other) -> bool:
        return isinstance(other, _HashableLossCfg) and self._key == other._key


class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.train.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.train_set = build_dataset(cfg.data, "train")
        self.val_set = build_dataset(cfg.data, "val")
        encoder = CenterNetTargetEncoder(
            tuple(cfg.data.input_size), cfg.data.num_classes, cfg.data.max_objects, cfg.model.output_stride
        )
        self.train_loader = DataLoader(self.train_set, cfg.data, train=True, target_fn=encoder, seed=cfg.train.seed)
        self.val_loader = DataLoader(self.val_set, cfg.data, train=False, target_fn=encoder)

        self.model = build_model(cfg)
        rng = jax.random.PRNGKey(cfg.train.seed)
        variables = init_variables(self.model, tuple(cfg.data.input_size), rng)
        if cfg.train.pretrained_backbone:
            variables, report = load_pretrained_backbone(variables, cfg.train.pretrained_backbone)
            (self.out_dir / "pretrained_report.json").write_text(json.dumps(report, indent=2))

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = max(1, self.steps_per_epoch * cfg.train.epochs)
        tx, self.schedule = build_optimizer(cfg.train, self.total_steps)
        self.state = create_train_state(self.model, variables, tx)
        self.loss_cfg = _HashableLossCfg(cfg.loss)
        self.backbone_bn_train = False if cfg.train.freeze_backbone_bn else None
        self.best_metric = -np.inf
        self.history: list[dict] = []
        cfg.save(self.out_dir / "config.yaml")

        log.info(
            "PerceptNet %s | %d classes | %.2fM params (%.0f%% backbone) | %d steps/epoch | devices=%s",
            cfg.model.backbone,
            cfg.data.num_classes,
            count_params(self.state.params) / 1e6,
            100 * frozen_fraction(self.state.params),
            self.steps_per_epoch,
            jax.devices(),
        )

    def train(self) -> dict:
        cfg = self.cfg.train
        step = int(self.state.step)
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            n = 0
            for batch in self.train_loader:
                self.state, metrics = train_step(
                    self.state, device_batch(batch), self.loss_cfg, self.backbone_bn_train
                )
                step += 1
                n += 1
                if step % cfg.log_every == 0 or step == 1:
                    m = {k: float(v) for k, v in metrics.items()}
                    log.info(
                        "epoch %d step %d/%d loss %.4f (hm %.4f size %.4f off %.4f) lr %.2e",
                        epoch, step, self.total_steps, m["loss"], m["heatmap_loss"],
                        m["size_loss"], m["offset_loss"], float(self.schedule(step)),
                    )
            dt = time.perf_counter() - t0
            record = {"epoch": epoch, "step": step, "epoch_time_s": dt,
                      "images_per_s": n * self.cfg.data.batch_size / max(dt, 1e-9)}
            if epoch % cfg.eval_every_epochs == 0 or epoch == cfg.epochs:
                record.update(self.evaluate())
                metric = record.get("map", -record.get("val_loss", np.inf))
                if metric > self.best_metric:
                    self.best_metric = metric
                    save_inference_variables(self.variables, self.out_dir / "best_model.msgpack")
            save_checkpoint(self.state, self.out_dir, "last", extra=record)
            self.history.append(record)
            log.info("epoch %d done: %s", epoch, json.dumps({k: round(v, 4) for k, v in record.items()}))
        save_inference_variables(self.variables, self.out_dir / "final_model.msgpack")
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        return self.history[-1] if self.history else {}

    @property
    def variables(self) -> dict:
        return {"params": self.state.params, "batch_stats": self.state.batch_stats}

    def evaluate(self) -> dict:
        """Validation loss averaged over the val split."""
        totals: dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            _, metrics = eval_step(self.state, device_batch(batch), self.loss_cfg)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n += 1
        return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}
