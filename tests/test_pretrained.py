import importlib.util
from pathlib import Path

import jax
import numpy as np

from perceptnet.config import Config
from perceptnet.model.detector import build_model, init_variables
from perceptnet.train.checkpoint import load_inference_variables, save_inference_variables
from perceptnet.train.pretrained import load_pretrained_backbone

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_torchvision_resnet.py"
_spec = importlib.util.spec_from_file_location("convert", _SCRIPT)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


def test_key_mapping():
    assert convert.map_key("conv1.weight") == ("params", "backbone/conv1/kernel")
    assert convert.map_key("bn1.running_var") == ("batch_stats", "backbone/bn1/var")
    assert convert.map_key("layer2.0.downsample.0.weight") == ("params", "backbone/layer2_0/downsample_conv/kernel")
    assert convert.map_key("layer2.0.downsample.1.weight") == ("params", "backbone/layer2_0/downsample_bn/scale")
    assert convert.map_key("layer3.1.bn2.bias") == ("params", "backbone/layer3_1/bn2/bias")
    assert convert.map_key("fc.weight") is None
    assert convert.map_key("bn1.num_batches_tracked") is None


def _variables():
    cfg = Config()
    cfg.data.input_size = (64, 64)
    model = build_model(cfg)
    return init_variables(model, (64, 64), jax.random.PRNGKey(0))


def test_converted_weights_load_into_backbone(tmp_path):
    variables = _variables()
    w = np.random.default_rng(0).normal(size=(64, 3, 7, 7)).astype(np.float32)  # OIHW
    converted = convert.convert({"conv1.weight": w, "bn1.running_mean": np.ones(64, np.float32)})
    path = tmp_path / "w.npz"
    np.savez(path, **converted)
    new, report = load_pretrained_backbone(variables, path)
    assert report["loaded"] == 2
    np.testing.assert_allclose(np.asarray(new["params"]["backbone"]["conv1"]["kernel"]), w.transpose(2, 3, 1, 0))
    np.testing.assert_allclose(np.asarray(new["batch_stats"]["backbone"]["bn1"]["mean"]), 1.0)


def test_inference_variables_roundtrip(tmp_path):
    variables = _variables()
    path = save_inference_variables(variables, tmp_path / "m.msgpack")
    loaded = load_inference_variables(path)
    a = jax.tree_util.tree_leaves(variables["params"])
    b = jax.tree_util.tree_leaves(loaded["params"])
    assert len(a) == len(b)
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]))
