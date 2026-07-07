import jax.numpy as jnp
import numpy as np
import pytest

from perceptnet.bench.ops_bench import clustered_boxes
from perceptnet.ops import cpp_ext
from perceptnet.ops.boxes import iou_matrix, iou_matrix_jax
from perceptnet.ops.nms import batched_nms_numpy, nms_jax, nms_numpy


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_iou_numpy_matches_jax():
    b, _, _ = clustered_boxes(200)
    np.testing.assert_allclose(iou_matrix(b, b), np.asarray(iou_matrix_jax(jnp.asarray(b), jnp.asarray(b))), atol=1e-5)


def test_nms_simple():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms_numpy(boxes, scores, 0.5).tolist() == [0, 2]


def test_nms_jax_matches_numpy():
    boxes, scores, _ = clustered_boxes(300, seed=3)
    ref = set(nms_numpy(boxes, scores, 0.5).tolist())
    order, keep = nms_jax(jnp.asarray(boxes), jnp.asarray(scores), 0.5)
    got = set(np.asarray(order)[np.asarray(keep)].tolist())
    assert got == ref


def test_batched_nms_keeps_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    labels = np.array([0, 1], np.int32)
    assert sorted(batched_nms_numpy(boxes, scores, labels, 0.5).tolist()) == [0, 1]


@pytest.mark.skipif(not cpp_ext.HAS_CPP, reason="C++ extension not built")
def test_cpp_matches_numpy():
    boxes, scores, labels = clustered_boxes(1000, seed=7)
    assert cpp_ext.nms(boxes, scores, 0.5, backend="cpp").tolist() == nms_numpy(boxes, scores, 0.5).tolist()
    assert set(cpp_ext.batched_nms(boxes, scores, labels, 0.5, backend="cpp").tolist()) == set(
        batched_nms_numpy(boxes, scores, labels, 0.5).tolist()
    )
    np.testing.assert_allclose(cpp_ext.iou_matrix(boxes, boxes, backend="cpp"), iou_matrix(boxes, boxes), atol=1e-5)


@pytest.mark.skipif(not cpp_ext.HAS_CPP_OPENCV, reason="C++ OpenCV kernels not built")
def test_cpp_preprocess_matches_python():
    frame = np.random.default_rng(0).integers(0, 255, (375, 1242, 3), dtype=np.uint8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    a, ia = cpp_ext.preprocess(frame, (384, 640), mean, std, backend="cpp")
    b, ib = cpp_ext.preprocess(frame, (384, 640), mean, std, backend="numpy")
    np.testing.assert_allclose(a, b, atol=1e-4)
    assert (ia.pad_x, ia.pad_y) == (ib.pad_x, ib.pad_y)
