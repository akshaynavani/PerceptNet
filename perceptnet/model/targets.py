"""CenterNet training-target encoding (NumPy, runs in the data loader threads)."""

from __future__ import annotations

import numpy as np


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Radius such that a box with corners shifted by it still has IoU >= ``min_overlap``.

    Follows the original CenterNet/CornerNet formulation (three corner-shift cases).
    """
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1**2 - 4 * a1 * c1)) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2**2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return float(min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """In-place element-wise max of a 2D gaussian (peak 1.0) into ``heatmap`` (H, W)."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)

    x, y = center
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return
    region = heatmap[y - top : y + bottom, x - left : x + right]
    g = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, g, out=region)


class CenterNetTargetEncoder:
    """Callable used as ``DataLoader(target_fn=...)``.

    Given boxes in input pixels and labels, produces:

    * ``heatmap`` (Ho, Wo, C) gaussian class heatmaps
    * ``indices`` (M,) flat ``y * Wo + x`` index of each object's center cell
    * ``size`` (M, 2) box (w, h) in output-stride units
    * ``offset`` (M, 2) sub-cell center offset in [0, 1)
    * ``reg_mask`` (M,) 1.0 for real objects
    """

    def __init__(self, input_hw: tuple[int, int], num_classes: int, max_objects: int, stride: int = 4):
        self.out_h, self.out_w = input_hw[0] // stride, input_hw[1] // stride
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.stride = stride

    def __call__(self, boxes: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        m, s = self.max_objects, self.stride
        heatmap = np.zeros((self.num_classes, self.out_h, self.out_w), dtype=np.float32)
        indices = np.zeros((m,), dtype=np.int32)
        size = np.zeros((m, 2), dtype=np.float32)
        offset = np.zeros((m, 2), dtype=np.float32)
        reg_mask = np.zeros((m,), dtype=np.float32)

        for k, (box, label) in enumerate(zip(boxes[:m], labels[:m])):
            bw, bh = (box[2] - box[0]) / s, (box[3] - box[1]) / s
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (box[0] + box[2]) / (2 * s), (box[1] + box[3]) / (2 * s)
            ix = int(np.clip(np.floor(cx), 0, self.out_w - 1))
            iy = int(np.clip(np.floor(cy), 0, self.out_h - 1))
            radius = max(0, int(gaussian_radius(np.ceil(bh), np.ceil(bw))))
            draw_gaussian(heatmap[int(label)], (ix, iy), radius)
            indices[k] = iy * self.out_w + ix
            size[k] = (bw, bh)
            offset[k] = (cx - ix, cy - iy)
            reg_mask[k] = 1.0

        return {
            "heatmap": np.ascontiguousarray(heatmap.transpose(1, 2, 0)),
            "indices": indices,
            "size": size,
            "offset": offset,
            "reg_mask": reg_mask,
        }
