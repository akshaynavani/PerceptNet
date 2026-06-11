"""Training-time augmentations, applied to letterboxed uint8 BGR images in input-pixel space."""

from __future__ import annotations

import cv2
import numpy as np

from perceptnet.data.preprocess import PAD_VALUE


def hflip(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = image.shape[1]
    flipped = image[:, ::-1].copy()
    out = boxes.copy()
    out[:, 0] = w - boxes[:, 2]
    out[:, 2] = w - boxes[:, 0]
    return flipped, out


def hsv_jitter(image: np.ndarray, gains: tuple[float, float, float], rng: np.random.Generator) -> np.ndarray:
    """YOLO-style hue/saturation/value jitter using an OpenCV LUT (fast, uint8 in/out)."""
    r = rng.uniform(-1, 1, 3) * np.asarray(gains) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_scale_translate(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scale_range: tuple[float, float],
    rng: np.random.Generator,
    min_visible: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zoom in/out around a random center and translate; boxes are clipped and filtered."""
    h, w = image.shape[:2]
    s = rng.uniform(*scale_range)
    # Keep the image center inside the canvas after transform.
    max_tx = 0.2 * w
    max_ty = 0.2 * h
    tx = rng.uniform(-max_tx, max_tx)
    ty = rng.uniform(-max_ty, max_ty)
    m = np.array(
        [[s, 0.0, (1 - s) * w / 2 + tx], [0.0, s, (1 - s) * h / 2 + ty]], dtype=np.float32
    )
    warped = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(PAD_VALUE,) * 3
    )
    if len(boxes) == 0:
        return warped, boxes, labels
    out = boxes.copy()
    out[:, [0, 2]] = boxes[:, [0, 2]] * s + m[0, 2]
    out[:, [1, 3]] = boxes[:, [1, 3]] * s + m[1, 2]
    area_before = (out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1])
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    area_after = (out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1])
    keep = (
        (area_after > min_visible * np.maximum(area_before, 1e-6))
        & ((out[:, 2] - out[:, 0]) >= 2)
        & ((out[:, 3] - out[:, 1]) >= 2)
    )
    return warped, out[keep], labels[keep]


class TrainAugment:
    def __init__(self, data_cfg, seed: int = 0) -> None:
        self.cfg = data_cfg
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, boxes, labels):
        cfg = self.cfg
        if cfg.scale_jitter and tuple(cfg.scale_jitter) != (1.0, 1.0):
            image, boxes, labels = random_scale_translate(
                image, boxes, labels, tuple(cfg.scale_jitter), self.rng
            )
        if self.rng.random() < cfg.hflip_prob:
            image, boxes = hflip(image, boxes)
        if any(g > 0 for g in cfg.hsv_jitter):
            image = hsv_jitter(image, tuple(cfg.hsv_jitter), self.rng)
        return image, boxes, labels
