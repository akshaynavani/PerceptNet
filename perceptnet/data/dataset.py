"""Dataset readers. Every dataset yields :class:`Sample` objects in original-image pixels."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Sample:
    image: np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)
    boxes: np.ndarray  # (N, 4) float32, xyxy in pixels
    labels: np.ndarray  # (N,) int32 contiguous class indices
    image_id: int


class DetectionDataset:
    """Minimal map-style dataset protocol."""

    class_names: list[str]

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Sample:  # pragma: no cover - interface
        raise NotImplementedError


class CocoDetection(DetectionDataset):
    """COCO-format detection dataset.

    KITTI, BDD100K, nuImages etc. are all trivially exportable to this format, so it is the
    single on-disk format PerceptNet supports. Category ids are remapped to contiguous
    ``0..C-1`` indices in ascending id order (or the order of ``class_names`` if given).
    """

    def __init__(
        self,
        images_dir: str | os.PathLike,
        annotation_file: str | os.PathLike,
        class_names: list[str] | None = None,
        min_box_size: float = 2.0,
        skip_empty: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        with open(annotation_file) as f:
            coco = json.load(f)

        categories = sorted(coco["categories"], key=lambda c: c["id"])
        if class_names:
            by_name = {c["name"]: c["id"] for c in categories}
            missing = [n for n in class_names if n not in by_name]
            if missing:
                raise ValueError(f"Classes {missing} not found in {annotation_file}")
            self.class_names = list(class_names)
            self.cat_to_label = {by_name[n]: i for i, n in enumerate(class_names)}
        else:
            self.class_names = [c["name"] for c in categories]
            self.cat_to_label = {c["id"]: i for i, c in enumerate(categories)}
        # COCO category id for each contiguous label (needed to write COCO-format results).
        self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}

        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.records: list[tuple[dict, np.ndarray, np.ndarray]] = []
        for img in coco["images"]:
            boxes, labels = [], []
            for ann in anns_by_image.get(img["id"], []):
                if ann.get("iscrowd", 0) or ann["category_id"] not in self.cat_to_label:
                    continue
                x, y, w, h = ann["bbox"]
                if w < min_box_size or h < min_box_size:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_to_label[ann["category_id"]])
            if skip_empty and not boxes:
                continue
            self.records.append(
                (
                    img,
                    np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    np.asarray(labels, dtype=np.int32),
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Sample:
        info, boxes, labels = self.records[idx]
        path = self.images_dir / info["file_name"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        return Sample(image=image, boxes=boxes.copy(), labels=labels.copy(), image_id=int(info["id"]))


def build_dataset(cfg, split: str) -> DetectionDataset:
    """Construct the train/val dataset described by a :class:`~perceptnet.config.DataConfig`."""
    from perceptnet.data.synthetic import SyntheticShapes

    if cfg.kind == "synthetic":
        size = cfg.synthetic_train_size if split == "train" else cfg.synthetic_val_size
        seed = 0 if split == "train" else 10_000_019
        return SyntheticShapes(
            num_samples=size,
            image_hw=tuple(cfg.input_size),
            class_names=cfg.class_names,
            seed=seed,
        )
    if cfg.kind == "coco":
        images = cfg.train_images if split == "train" else cfg.val_images
        anns = cfg.train_annotations if split == "train" else cfg.val_annotations
        ds = CocoDetection(images, anns, class_names=cfg.class_names or None)
        if not cfg.class_names:
            cfg.class_names = ds.class_names
        return ds
    raise ValueError(f"Unknown dataset kind '{cfg.kind}'")
