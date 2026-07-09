import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perceptnet.config import Config, load_config
from perceptnet.model.detector import build_model, count_params, init_variables
from perceptnet.train.optim import build_optimizer, param_groups


def small_cfg() -> Config:
    cfg = Config()
    cfg.data.input_size = (64, 96)
    cfg.data.class_names = ["car", "pedestrian"]
    return cfg


def test_output_shapes():
    cfg = small_cfg()
    model = build_model(cfg)
    variables = init_variables(model, cfg.data.input_size, jax.random.PRNGKey(0))
    out = model.apply(variables, jnp.zeros((2, 64, 96, 3)), train=False)
    assert out["heatmap"].shape == (2, 16, 24, 2)
    assert out["size"].shape == (2, 16, 24, 2)
    assert out["offset"].shape == (2, 16, 24, 2)
    assert out["heatmap"].dtype == jnp.float32
    assert 10e6 < count_params(variables["params"]) < 20e6  # ResNet-18 + neck + heads


def test_heatmap_prior_bias():
    cfg = small_cfg()
    model = build_model(cfg)
    variables = init_variables(model, cfg.data.input_size, jax.random.PRNGKey(0))
    out = model.apply(variables, jnp.zeros((1, 64, 96, 3)), train=False)
    p = jax.nn.sigmoid(out["heatmap"]).mean()
    assert abs(float(p) - cfg.model.heatmap_prior) < 0.02


def test_rejects_bad_input_size():
    cfg = small_cfg()
    cfg.data.input_size = (65, 96)
    with pytest.raises(ValueError):
        build_model(cfg)


def test_backbone_frozen_for_warmup_steps():
    cfg = small_cfg()
    cfg.train.freeze_backbone_steps = 5
    model = build_model(cfg)
    params = init_variables(model, cfg.data.input_size, jax.random.PRNGKey(0))["params"]
    tx, _ = build_optimizer(cfg.train, total_steps=100)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, opt_state, params)
    groups = param_groups(params)
    bb = jax.tree_util.tree_leaves(updates["backbone"])
    assert all(float(jnp.abs(u).max()) == 0.0 for u in bb)
    assert groups["heads"]["heatmap"]["out"]["kernel"] == "head"


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  input_size: [128, 256]\ntrain:\n  epochs: 3\n")
    cfg = load_config(str(path), ["train.lr=0.01", "inference.use_nms=false"])
    assert cfg.data.input_size == (128, 256)
    assert cfg.train.epochs == 3 and cfg.train.lr == 0.01
    assert cfg.inference.use_nms is False
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])


def test_numpy_dtype_roundtrip():
    assert np.float32 == jnp.float32
