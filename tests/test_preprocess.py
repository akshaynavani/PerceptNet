import numpy as np
import pytest

from perceptnet.data.preprocess import (
    PAD_VALUE,
    boxes_to_input,
    boxes_to_original,
    denormalize,
    letterbox,
    normalize,
    preprocess_frame,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.mark.parametrize("hw", [(720, 1280), (480, 640), (100, 300), (300, 100)])
def test_letterbox_shape_and_padding(hw):
    img = np.full((*hw, 3), 200, np.uint8)
    out, info = letterbox(img, (256, 384))
    assert out.shape == (256, 384, 3)
    assert info.orig_h, info.orig_w == hw
    # Padding (if any) uses the constant pad value.
    if info.pad_y >= 1:
        assert (out[0] == PAD_VALUE).all()
    if info.pad_x >= 1:
        assert (out[:, 0] == PAD_VALUE).all()


def test_box_roundtrip():
    img = np.zeros((375, 1242, 3), np.uint8)
    _, info = letterbox(img, (384, 640))
    boxes = np.array([[10, 20, 300, 200], [0, 0, 1242, 375]], np.float32)
    back = boxes_to_original(boxes_to_input(boxes, info), info)
    np.testing.assert_allclose(back, boxes, atol=1e-3)


def test_normalize_inverse():
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    np.testing.assert_allclose(denormalize(normalize(rgb, MEAN, STD), MEAN, STD), rgb, atol=1)


def test_preprocess_frame_is_rgb():
    bgr = np.zeros((32, 32, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    out, _ = preprocess_frame(bgr, (32, 32), (0, 0, 0), (1, 1, 1))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 1.0])
