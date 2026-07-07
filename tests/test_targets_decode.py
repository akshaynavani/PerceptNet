import jax.numpy as jnp
import numpy as np

from perceptnet.model.decode import decode
from perceptnet.model.losses import focal_loss, gather_feature
from perceptnet.model.targets import CenterNetTargetEncoder, draw_gaussian, gaussian_radius


def test_gaussian_peak_and_clipping():
    hm = np.zeros((10, 10), np.float32)
    draw_gaussian(hm, (0, 0), 3)  # corner: must clip without error
    assert hm[0, 0] == 1.0
    assert hm.max() == 1.0
    assert gaussian_radius(10, 10) > 0


def test_encoder_outputs():
    enc = CenterNetTargetEncoder((128, 128), num_classes=3, max_objects=4)
    boxes = np.array([[10, 10, 50, 40], [60, 64, 100, 120]], np.float32)
    labels = np.array([0, 2], np.int32)
    t = enc(boxes, labels)
    assert t["heatmap"].shape == (32, 32, 3)
    assert t["reg_mask"].tolist() == [1, 1, 0, 0]
    # Peak of class 0 is at the floor of the stride-4 center.
    cx, cy = (10 + 50) / 8, (10 + 40) / 8
    assert t["heatmap"][int(cy), int(cx), 0] == 1.0
    np.testing.assert_allclose(t["size"][0], [10.0, 7.5])
    np.testing.assert_allclose(t["offset"][0], [cx - int(cx), cy - int(cy)], atol=1e-6)


def test_decode_recovers_encoded_boxes():
    """Feeding perfect targets as network outputs must decode back to the GT boxes."""
    enc = CenterNetTargetEncoder((128, 160), num_classes=2, max_objects=8)
    boxes = np.array([[8, 12, 48, 60], [90, 30, 150, 70]], np.float32)
    labels = np.array([1, 0], np.int32)
    t = enc(boxes, labels)
    ho, wo = t["heatmap"].shape[:2]
    hm = np.clip(t["heatmap"], 1e-6, 1 - 1e-6)
    logits = np.log(hm / (1 - hm))
    size = np.zeros((ho * wo, 2), np.float32)
    off = np.zeros((ho * wo, 2), np.float32)
    for k in range(2):
        size[t["indices"][k]] = t["size"][k]
        off[t["indices"][k]] = t["offset"][k]
    outputs = {
        "heatmap": jnp.asarray(logits[None]),
        "size": jnp.asarray(size.reshape(1, ho, wo, 2)),
        "offset": jnp.asarray(off.reshape(1, ho, wo, 2)),
    }
    pred_boxes, scores, classes = decode(outputs, top_k=2, stride=4)
    pred_boxes, classes = np.asarray(pred_boxes[0]), np.asarray(classes[0])
    for box, label in zip(boxes, labels):
        i = int(np.argmin(np.abs(pred_boxes - box).sum(1)))
        np.testing.assert_allclose(pred_boxes[i], box, atol=1e-3)
        assert classes[i] == label


def test_gather_feature():
    fm = jnp.arange(2 * 3 * 4 * 1, dtype=jnp.float32).reshape(2, 3, 4, 1)
    idx = jnp.array([[0, 5], [11, 1]])
    np.testing.assert_allclose(np.asarray(gather_feature(fm, idx))[..., 0], [[0, 5], [23, 13]])


def test_focal_loss_prefers_correct_predictions():
    target = np.zeros((1, 8, 8, 1), np.float32)
    target[0, 4, 4, 0] = 1.0
    good = np.full_like(target, -6.0)
    good[0, 4, 4, 0] = 6.0
    bad = -good
    assert float(focal_loss(jnp.asarray(good), jnp.asarray(target))) < float(
        focal_loss(jnp.asarray(bad), jnp.asarray(target))
    )
