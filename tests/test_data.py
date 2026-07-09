import json

import cv2
import numpy as np

from perceptnet.config import Config
from perceptnet.data.augment import hflip, random_scale_translate
from perceptnet.data.dataset import CocoDetection
from perceptnet.data.loader import DataLoader
from perceptnet.data.synthetic import SyntheticShapes, export_coco
from perceptnet.model.targets import CenterNetTargetEncoder


def test_synthetic_is_deterministic():
    ds = SyntheticShapes(5, (128, 128), seed=1)
    a, b = ds[3], ds[3]
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.boxes, b.boxes)
    assert len(a.boxes) == len(a.labels) >= 1


def test_coco_roundtrip(tmp_path):
    ds = SyntheticShapes(4, (96, 128), seed=2)
    ann = export_coco(ds, str(tmp_path))
    coco = CocoDetection(tmp_path / "images", ann)
    assert len(coco) == 4
    assert coco.class_names == ds.class_names
    s = coco[1]
    np.testing.assert_allclose(s.boxes, ds[1].boxes, atol=1e-4)
    np.testing.assert_array_equal(s.labels, ds[1].labels)


def test_coco_class_filter(tmp_path):
    img = np.zeros((32, 32, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    data = {
        "images": [{"id": 7, "file_name": "a.png"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 1, 10, 10]},
            {"id": 2, "image_id": 7, "category_id": 5, "bbox": [1, 1, 10, 10]},
            {"id": 3, "image_id": 7, "category_id": 3, "bbox": [1, 1, 10, 10], "iscrowd": 1},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 5, "name": "truck"}],
    }
    (tmp_path / "ann.json").write_text(json.dumps(data))
    ds = CocoDetection(tmp_path, tmp_path / "ann.json", class_names=["truck"])
    s = ds[0]
    assert s.labels.tolist() == [0] and s.image_id == 7


def test_loader_fixed_shapes_and_padding():
    cfg = Config()
    cfg.data.input_size = (64, 64)
    cfg.data.batch_size = 4
    cfg.data.max_objects = 16
    cfg.data.num_workers = 2
    ds = SyntheticShapes(6, (80, 120), seed=0)
    enc = CenterNetTargetEncoder((64, 64), 3, 16)
    loader = DataLoader(ds, cfg.data, train=False, target_fn=enc)
    batches = list(loader)
    assert len(batches) == len(loader) == 2
    for b in batches:
        assert b["image"].shape == (4, 64, 64, 3)
        assert b["heatmap"].shape == (4, 16, 16, 3)
        assert b["gt_boxes"].shape == (4, 16, 4)
    assert batches[-1]["valid"].tolist() == [True, True, False, False]
    loader.close()


def test_hflip_boxes():
    img = np.zeros((10, 20, 3), np.uint8)
    _, boxes = hflip(img, np.array([[2, 1, 5, 4]], np.float32))
    np.testing.assert_allclose(boxes, [[15, 1, 18, 4]])


def test_scale_translate_keeps_boxes_inside():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64, 3), np.uint8)
    boxes = np.array([[10, 10, 30, 30], [40, 40, 60, 60]], np.float32)
    for _ in range(20):
        _, out, labels = random_scale_translate(img, boxes, np.array([0, 1]), (0.5, 1.5), rng)
        assert len(out) == len(labels)
        assert (out >= 0).all() and (out[:, [0, 2]] <= 64).all() and (out[:, [1, 3]] <= 64).all()
