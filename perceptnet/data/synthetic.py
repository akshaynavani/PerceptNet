"""Procedurally generated "driving-like" shapes dataset for smoke tests and CI.

Each class is rendered as a distinct shape family on a textured road-ish background, so a
detector can genuinely learn it in a few epochs on CPU. Samples are deterministic in
``(seed, index)`` which makes evaluation reproducible.
"""

from __future__ import annotations

import cv2
import numpy as np

from perceptnet.data.dataset import DetectionDataset, Sample

_PALETTE = [
    (40, 40, 200),
    (200, 160, 30),
    (40, 180, 60),
    (180, 60, 180),
    (30, 200, 220),
    (220, 220, 220),
]


class SyntheticShapes(DetectionDataset):
    def __init__(
        self,
        num_samples: int,
        image_hw: tuple[int, int] = (256, 256),
        class_names: list[str] | None = None,
        max_objects: int = 8,
        min_size: int = 12,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.image_hw = image_hw
        self.class_names = list(class_names or ["car", "pedestrian", "cyclist"])
        self.max_objects = max_objects
        self.min_size = min_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _background(self, rng: np.random.Generator) -> np.ndarray:
        h, w = self.image_hw
        sky = rng.integers(120, 200)
        road = rng.integers(50, 100)
        horizon = int(h * rng.uniform(0.3, 0.5))
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:horizon] = (sky, sky - 10, sky - 40)
        img[horizon:] = (road, road, road)
        noise = rng.normal(0, 8, size=img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _draw(self, img: np.ndarray, label: int, box: np.ndarray, color) -> None:
        x1, y1, x2, y2 = box.astype(int)
        shape = label % 3
        if shape == 0:  # filled rectangle ("car")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), thickness=1)
        elif shape == 1:  # ellipse ("pedestrian")
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness=-1)
        else:  # triangle ("cyclist")
            pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
        # Additional families for >3 classes get a distinguishing inner mark.
        if label >= 3:
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), max(2, (x2 - x1) // 6), (255, 255, 255), -1)

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)
        rng = np.random.default_rng((self.seed, idx))
        h, w = self.image_hw
        img = self._background(rng)
        n = int(rng.integers(1, self.max_objects + 1))
        boxes, labels = [], []
        for _ in range(n):
            label = int(rng.integers(0, len(self.class_names)))
            shape = label % 3
            bw = int(rng.integers(self.min_size, max(self.min_size + 1, w // 3)))
            aspect = {0: rng.uniform(0.5, 0.9), 1: rng.uniform(1.8, 2.8), 2: rng.uniform(1.0, 1.4)}[shape]
            bh = int(np.clip(bw * aspect, self.min_size, h // 2))
            x1 = int(rng.integers(0, max(1, w - bw)))
            y1 = int(rng.integers(0, max(1, h - bh)))
            box = np.array([x1, y1, x1 + bw, y1 + bh], dtype=np.float32)
            # Avoid heavy overlap so ground truth stays unambiguous.
            if boxes and _max_iou(box, np.asarray(boxes)) > 0.3:
                continue
            color = _PALETTE[label % len(_PALETTE)]
            jitter = rng.integers(-25, 26, size=3)
            color = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(color, jitter))
            self._draw(img, label, box, color)
            boxes.append(box)
            labels.append(label)
        return Sample(
            image=img,
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels, dtype=np.int32),
            image_id=idx,
        )


def _max_iou(box: np.ndarray, others: np.ndarray) -> float:
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return float(np.max(inter / (area + areas - inter + 1e-9)))


def export_coco(dataset: SyntheticShapes, out_dir: str) -> str:
    """Write a synthetic split to disk in COCO format (images + annotations JSON)."""
    import json
    from pathlib import Path

    root = Path(out_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i in range(len(dataset)):
        s = dataset[i]
        name = f"{i:06d}.png"
        cv2.imwrite(str(root / "images" / name), s.image)
        images.append({"id": s.image_id, "file_name": name, "height": s.image.shape[0], "width": s.image.shape[1]})
        for box, label in zip(s.boxes, s.labels):
            x1, y1, x2, y2 = box.tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": s.image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(dataset.class_names)]
    path = root / "annotations.json"
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return str(path)
