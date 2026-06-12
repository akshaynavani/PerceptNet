"""Batched, fixed-shape data loading.

XLA compiles one executable per input shape, so every batch emitted here has identical
shapes: images are letterboxed to ``input_size``, ground truth is padded to ``max_objects``,
and a short final batch is padded with dummy samples flagged by ``valid=False``.
Sample decoding runs in a thread pool (OpenCV releases the GIL).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from perceptnet.data.augment import TrainAugment
from perceptnet.data.dataset import DetectionDataset
from perceptnet.data.preprocess import LetterboxInfo, boxes_to_input, letterbox, normalize

Batch = dict[str, np.ndarray]


class DataLoader:
    def __init__(
        self,
        dataset: DetectionDataset,
        data_cfg,
        train: bool,
        target_fn: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]] | None = None,
        seed: int = 0,
        drop_last: bool | None = None,
    ) -> None:
        self.dataset = dataset
        self.cfg = data_cfg
        self.train = train
        self.target_fn = target_fn
        self.batch_size = data_cfg.batch_size
        self.input_hw = tuple(data_cfg.input_size)
        self.max_objects = data_cfg.max_objects
        self.drop_last = train if drop_last is None else drop_last
        self.rng = np.random.default_rng(seed)
        self.augment = TrainAugment(data_cfg, seed=seed + 1) if train else None
        self._pool = ThreadPoolExecutor(max_workers=max(1, data_cfg.num_workers))
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _load_one(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, LetterboxInfo, int]:
        sample = self.dataset[idx]
        image, info = letterbox(sample.image, self.input_hw)
        boxes = boxes_to_input(sample.boxes, info) if len(sample.boxes) else sample.boxes
        labels = sample.labels
        if self.augment is not None:
            image, boxes, labels = self.augment(image, boxes, labels)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return normalize(rgb, self.cfg.mean, self.cfg.std), boxes, labels, info, sample.image_id

    def _collate(self, items, n_valid: int) -> Batch:
        b, (h, w), m = self.batch_size, self.input_hw, self.max_objects
        images = np.zeros((b, h, w, 3), dtype=np.float32)
        gt_boxes = np.zeros((b, m, 4), dtype=np.float32)
        gt_labels = np.full((b, m), -1, dtype=np.int32)
        valid = np.zeros((b,), dtype=bool)
        image_ids = np.full((b,), -1, dtype=np.int64)
        lb = np.zeros((b, 5), dtype=np.float32)  # scale, pad_x, pad_y, orig_h, orig_w
        targets: dict[str, list[np.ndarray]] = {}
        for i, (img, boxes, labels, info, image_id) in enumerate(items):
            images[i] = img
            k = min(len(boxes), m)
            gt_boxes[i, :k] = boxes[:k]
            gt_labels[i, :k] = labels[:k]
            valid[i] = i < n_valid
            image_ids[i] = image_id
            lb[i] = (info.scale, info.pad_x, info.pad_y, info.orig_h, info.orig_w)
            if self.target_fn is not None:
                for key, val in self.target_fn(boxes[:k], labels[:k]).items():
                    targets.setdefault(key, []).append(val)
        batch: Batch = {
            "image": images,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "valid": valid,
            "image_id": image_ids,
            "letterbox": lb,
        }
        for key, vals in targets.items():
            batch[key] = np.stack(vals)
        return batch

    def __iter__(self) -> Iterator[Batch]:
        n = len(self.dataset)
        order = np.arange(n)
        if self.train and self.cfg.shuffle:
            self.rng.shuffle(order)
        self.epoch += 1
        for start in range(0, n, self.batch_size):
            idxs = order[start : start + self.batch_size]
            n_valid = len(idxs)
            if n_valid < self.batch_size:
                if self.drop_last:
                    return
                # Pad with repeats of the first index; they are masked out via ``valid``.
                idxs = np.concatenate([idxs, np.full(self.batch_size - n_valid, idxs[0])])
            items = list(self._pool.map(self._load_one, idxs.tolist()))
            yield self._collate(items, n_valid)

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def letterbox_infos(batch: Batch) -> list[LetterboxInfo]:
    return [
        LetterboxInfo(float(s), float(px), float(py), int(h), int(w))
        for s, px, py, h, w in batch["letterbox"]
    ]
