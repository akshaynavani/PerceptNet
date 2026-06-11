"""OpenCV frame preprocessing shared by training and inference.

The same letterbox + normalize path is used at train time and deploy time so the network
never sees a distribution shift caused by preprocessing differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """Parameters needed to map boxes between original-frame and network-input coordinates."""

    scale: float
    pad_x: float
    pad_y: float
    orig_h: int
    orig_w: int


def letterbox(
    image: np.ndarray,
    out_hw: tuple[int, int],
    pad_value: int = PAD_VALUE,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` keeping aspect ratio and pad to ``out_hw`` (centered).

    Returns the padded uint8 image and the transform parameters.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {image.shape}")
    h, w = image.shape[:2]
    out_h, out_w = out_hw
    scale = min(out_h / h, out_w / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if (new_h, new_w) != (h, w):
        interp = cv2.INTER_AREA if scale < 1.0 else interpolation
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    else:
        resized = image
    pad_y = (out_h - new_h) / 2.0
    pad_x = (out_w - new_w) / 2.0
    top, left = int(np.floor(pad_y)), int(np.floor(pad_x))
    bottom, right = out_h - new_h - top, out_w - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    return padded, LetterboxInfo(scale, float(left), float(top), h, w)


def normalize(
    image_rgb_u8: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """uint8 RGB HxWx3 -> float32 normalized HxWx3 (NHWC is the native JAX/XLA conv layout)."""
    img = image_rgb_u8.astype(np.float32) * (1.0 / 255.0)
    img -= np.asarray(mean, dtype=np.float32)
    img /= np.asarray(std, dtype=np.float32)
    return img


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_hw: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[np.ndarray, LetterboxInfo]:
    """Full deploy-time preprocessing for one BGR frame (as read by ``cv2.imread``/``VideoCapture``)."""
    boxed, info = letterbox(frame_bgr, input_hw)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return normalize(rgb, mean, std), info


def preprocess_batch(
    frames_bgr: list[np.ndarray],
    input_hw: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[np.ndarray, list[LetterboxInfo]]:
    out = np.empty((len(frames_bgr), input_hw[0], input_hw[1], 3), dtype=np.float32)
    infos = []
    for i, frame in enumerate(frames_bgr):
        out[i], info = preprocess_frame(frame, input_hw, mean, std)
        infos.append(info)
    return out, infos


def boxes_to_input(boxes_xyxy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map boxes from original-frame pixels to letterboxed network-input pixels."""
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * info.scale + info.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * info.scale + info.pad_y
    return out


def boxes_to_original(boxes_xyxy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Inverse of :func:`boxes_to_input`, clipped to the original frame."""
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, info.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, info.orig_h)
    return out


def denormalize(image: np.ndarray, mean, std) -> np.ndarray:
    """Inverse of :func:`normalize`, returns uint8 RGB (useful for visualizing batches)."""
    img = image * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
