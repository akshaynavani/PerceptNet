"""OpenCV drawing helpers for detections."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np


def class_colors(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.8, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def draw_detections(frame_bgr: np.ndarray, detections, class_names: list[str], fps: float | None = None) -> np.ndarray:
    out = frame_bgr.copy()
    colors = class_colors(len(class_names))
    for box, score, label in zip(detections.boxes, detections.scores, detections.labels):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = colors[int(label) % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        text = f"{name} {score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - base - 2)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, max(th, y1 - base - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out
