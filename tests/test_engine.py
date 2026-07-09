import jax
import numpy as np
import pytest

from perceptnet.config import Config
from perceptnet.inference.engine import Detections, InferenceEngine
from perceptnet.model.detector import build_model, init_variables


@pytest.fixture(scope="module")
def engine():
    cfg = Config()
    cfg.data.input_size = (64, 96)
    cfg.inference.score_threshold = 0.0
    cfg.inference.top_k = 20
    cfg.inference.batch_size = 2
    variables = init_variables(build_model(cfg), (64, 96), jax.random.PRNGKey(0))
    return InferenceEngine(cfg, variables)


def test_detect_returns_original_coordinates(engine):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (120, 200, 3), dtype=np.uint8) for _ in range(3)]
    dets = engine.detect(frames)
    assert len(dets) == 3 and all(isinstance(d, Detections) for d in dets)
    for d in dets:
        assert d.boxes.shape[1] == 4
        assert (d.boxes[:, [0, 2]] <= 200).all() and (d.boxes[:, [1, 3]] <= 120).all()
        assert (np.diff(d.scores) <= 1e-6).all() or len(d) <= 1
    assert engine.timings.summary()["device_ms_p50"] > 0


def test_infer_batch_shape_check(engine):
    with pytest.raises(ValueError):
        engine.infer_batch(np.zeros((1, 64, 96, 3), np.float32))
    boxes, scores, classes = engine.infer_batch(np.zeros((2, 64, 96, 3), np.float32))
    assert boxes.shape == (2, 20, 4) and scores.shape == (2, 20) and classes.dtype == np.int32


def test_hlo_dump(engine, tmp_path):
    paths = engine.dump_hlo(tmp_path)
    assert paths[0].read_text().strip()
    assert engine.compile_time_s > 0
