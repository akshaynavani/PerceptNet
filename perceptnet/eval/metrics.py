"""Detection accuracy metrics.

* COCO-style AP: greedy score-ordered matching per image, 101-point interpolated precision,
  averaged over IoU thresholds 0.50:0.05:0.95 (AP), plus AP50 / AP75 and per-class AP.
* Score-threshold sweep: precision / recall / F1 at IoU 0.5 for a range of confidence
  thresholds — the operating-point view needed to pick a deployment threshold.

Pure NumPy; no pycocotools dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perceptnet.ops.boxes import iou_matrix

COCO_IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class _ImageRecord:
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_labels: np.ndarray
    gt_boxes: np.ndarray
    gt_labels: np.ndarray


def match_detections(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float,
    ious: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy matching for a single image and class.

    Predictions are visited in descending score order; each takes the unmatched ground truth with
    the highest IoU >= ``iou_threshold``. Returns a boolean TP flag per prediction (input order).
    """
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    if ious is None:
        ious = iou_matrix(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred_scores, kind="stable"):
        cand = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_threshold:
            taken[j] = True
            tp[i] = True
    return tp


def average_precision(tp_sorted: np.ndarray, num_gt: int) -> float:
    """101-point interpolated AP from TP flags sorted by descending score."""
    if num_gt == 0:
        return float("nan")
    if len(tp_sorted) == 0:
        return 0.0
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(~tp_sorted)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Precision envelope (monotonically non-increasing from the right).
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass
class DetectionEvaluator:
    num_classes: int
    class_names: list[str] | None = None
    iou_thresholds: np.ndarray = field(default_factory=lambda: COCO_IOU_THRESHOLDS.copy())
    score_thresholds: np.ndarray = field(default_factory=lambda: np.round(np.arange(0.05, 1.0, 0.05), 2))
    _images: list[_ImageRecord] = field(default_factory=list)

    def add(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels) -> None:
        self._images.append(
            _ImageRecord(
                np.asarray(pred_boxes, np.float32).reshape(-1, 4),
                np.asarray(pred_scores, np.float32).reshape(-1),
                np.asarray(pred_labels, np.int64).reshape(-1),
                np.asarray(gt_boxes, np.float32).reshape(-1, 4),
                np.asarray(gt_labels, np.int64).reshape(-1),
            )
        )

    def reset(self) -> None:
        self._images.clear()

    def _per_class(self, c: int):
        """Scores, per-IoU-threshold TP flags (both sorted by score) and #GT for class ``c``."""
        scores, tps, num_gt = [], [], 0
        for rec in self._images:
            pm = rec.pred_labels == c
            gm = rec.gt_labels == c
            pb, ps, gb = rec.pred_boxes[pm], rec.pred_scores[pm], rec.gt_boxes[gm]
            num_gt += len(gb)
            if len(pb) == 0:
                continue
            ious = iou_matrix(pb, gb) if len(gb) else None
            tps.append(np.stack([match_detections(pb, ps, gb, t, ious) for t in self.iou_thresholds], axis=0))
            scores.append(ps)
        if not scores:
            return np.zeros(0), np.zeros((len(self.iou_thresholds), 0), bool), num_gt
        scores = np.concatenate(scores)
        tps = np.concatenate(tps, axis=1)
        order = np.argsort(-scores, kind="stable")
        return scores[order], tps[:, order], num_gt

    def compute(self) -> dict:
        names = self.class_names or [str(i) for i in range(self.num_classes)]
        iou_t = list(self.iou_thresholds)
        i50 = iou_t.index(0.5) if 0.5 in iou_t else 0
        i75 = iou_t.index(0.75) if 0.75 in iou_t else None

        ap = np.full((self.num_classes, len(iou_t)), np.nan)
        sweep_tp = np.zeros(len(self.score_thresholds))
        sweep_pred = np.zeros(len(self.score_thresholds))
        total_gt = 0
        per_class = {}
        for c in range(self.num_classes):
            scores, tps, num_gt = self._per_class(c)
            total_gt += num_gt
            for t in range(len(iou_t)):
                ap[c, t] = average_precision(tps[t], num_gt)
            for k, thr in enumerate(self.score_thresholds):
                sel = scores >= thr
                sweep_tp[k] += tps[i50][sel].sum()
                sweep_pred[k] += sel.sum()
            per_class[names[c]] = {
                "ap": _nanmean(ap[c]),
                "ap50": _nan_to_none(ap[c, i50]),
                "num_gt": int(num_gt),
                "num_pred": int(len(scores)),
            }

        precision = sweep_tp / np.maximum(sweep_pred, 1)
        recall = sweep_tp / max(total_gt, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best = int(np.argmax(f1)) if len(f1) else 0
        return {
            "map": _nanmean(ap),
            "map50": _nanmean(ap[:, i50]),
            "map75": _nanmean(ap[:, i75]) if i75 is not None else None,
            "ap_per_iou": {f"{t:.2f}": _nanmean(ap[:, k]) for k, t in enumerate(iou_t)},
            "per_class": per_class,
            "threshold_sweep": [
                {"score_threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}
                for t, p, r, f in zip(self.score_thresholds, precision, recall, f1)
            ],
            "best_f1": {
                "score_threshold": float(self.score_thresholds[best]) if len(f1) else None,
                "f1": float(f1[best]) if len(f1) else 0.0,
            },
            "num_images": len(self._images),
            "num_gt": int(total_gt),
        }


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else 0.0


def _nan_to_none(x):
    return None if np.isnan(x) else float(x)
