import numpy as np

from perceptnet.eval.metrics import DetectionEvaluator, average_precision, match_detections


def test_perfect_predictions_give_map_one():
    ev = DetectionEvaluator(num_classes=2, class_names=["a", "b"])
    gt = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], np.float32)
    labels = np.array([0, 1])
    ev.add(gt, [0.9, 0.8], labels, gt, labels)
    r = ev.compute()
    assert abs(r["map"] - 1.0) < 1e-9
    assert abs(r["map50"] - 1.0) < 1e-9
    assert r["per_class"]["a"]["num_gt"] == 1


def test_false_positive_and_miss_lower_ap():
    ev = DetectionEvaluator(num_classes=1)
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    preds = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    ev.add(preds, [0.9, 0.95], [0, 0], gt, [0, 0])
    r = ev.compute()
    assert 0.0 < r["map50"] < 1.0


def test_match_each_gt_once():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    tp = match_detections(preds, np.array([0.9, 0.8]), gt, 0.5)
    assert tp.tolist() == [True, False]


def test_average_precision_edge_cases():
    assert np.isnan(average_precision(np.array([], bool), 0))
    assert average_precision(np.array([], bool), 3) == 0.0
    assert average_precision(np.array([True, True]), 2) == 1.0


def test_threshold_sweep_monotone_recall():
    rng = np.random.default_rng(0)
    ev = DetectionEvaluator(num_classes=1)
    for _ in range(20):
        gt = rng.uniform(0, 100, (3, 2))
        gt = np.concatenate([gt, gt + 20], 1).astype(np.float32)
        noisy = gt + rng.normal(0, 3, gt.shape).astype(np.float32)
        ev.add(noisy, rng.uniform(0, 1, 3), [0, 0, 0], gt, [0, 0, 0])
    r = ev.compute()
    recalls = [row["recall"] for row in r["threshold_sweep"]]
    assert all(a >= b - 1e-12 for a, b in zip(recalls, recalls[1:]))
    assert 0.0 <= r["best_f1"]["f1"] <= 1.0
