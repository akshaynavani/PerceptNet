"""Run a detector over a validation split and compute accuracy metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from perceptnet.config import Config
from perceptnet.data.dataset import build_dataset
from perceptnet.data.loader import DataLoader, letterbox_infos
from perceptnet.data.preprocess import boxes_to_original
from perceptnet.eval.metrics import DetectionEvaluator
from perceptnet.inference.engine import InferenceEngine, postprocess_detections

log = logging.getLogger(__name__)

# mAP needs the full precision/recall curve, so evaluation keeps low-confidence detections.
EVAL_SCORE_THRESHOLD = 0.01


def accumulate_batch(
    evaluator: DetectionEvaluator,
    batch: dict,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    inference_cfg,
    score_threshold: float = EVAL_SCORE_THRESHOLD,
) -> None:
    """Add one batch of decoded top-K outputs (network-input pixels) to ``evaluator``."""
    for i, info in enumerate(letterbox_infos(batch)):
        if not batch["valid"][i]:
            continue
        dets = postprocess_detections(boxes[i], scores[i], classes[i], info, inference_cfg, score_threshold)
        gt_mask = batch["gt_labels"][i] >= 0
        gt_boxes = boxes_to_original(batch["gt_boxes"][i][gt_mask], info)
        evaluator.add(dets.boxes, dets.scores, dets.labels, gt_boxes, batch["gt_labels"][i][gt_mask])


def run_evaluation(cfg: Config, checkpoint: str | Path, split: str = "val", out: str | Path | None = None) -> dict:
    dataset = build_dataset(cfg.data, split)
    loader = DataLoader(dataset, cfg.data, train=False)
    engine = InferenceEngine.from_checkpoint(cfg, checkpoint, batch_size=cfg.data.batch_size)
    evaluator = DetectionEvaluator(cfg.data.num_classes, cfg.data.class_names)
    for batch in loader:
        boxes, scores, classes = engine.infer_batch(batch["image"])
        accumulate_batch(evaluator, batch, boxes, scores, classes, cfg.inference)
    loader.close()
    report = evaluator.compute()
    report["compile_time_s"] = engine.compile_time_s
    log.info("mAP %.4f | mAP50 %.4f | mAP75 %.4f", report["map"], report["map50"], report["map75"] or 0.0)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = [
        f"images: {report['num_images']}  ground truth: {report['num_gt']}",
        f"mAP@[.5:.95] {report['map']:.4f}   mAP50 {report['map50']:.4f}   mAP75 {(report['map75'] or 0):.4f}",
        "",
        f"{'class':<16}{'AP':>8}{'AP50':>8}{'#gt':>8}{'#pred':>8}",
    ]
    for name, row in report["per_class"].items():
        ap50 = "-" if row["ap50"] is None else f"{row['ap50']:.4f}"
        lines.append(f"{name:<16}{row['ap']:>8.4f}{ap50:>8}{row['num_gt']:>8}{row['num_pred']:>8}")
    lines += ["", f"{'score_thr':>10}{'precision':>11}{'recall':>9}{'f1':>8}"]
    for row in report["threshold_sweep"]:
        lines.append(
            f"{row['score_threshold']:>10.2f}{row['precision']:>11.4f}{row['recall']:>9.4f}{row['f1']:>8.4f}"
        )
    best = report["best_f1"]
    lines.append(f"\nbest F1 {best['f1']:.4f} at score threshold {best['score_threshold']}")
    return "\n".join(lines)
