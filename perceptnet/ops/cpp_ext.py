"""Dispatch layer for the optional C++ extension with NumPy fallbacks.

Build the extension with::

    cmake -S cpp -B build && cmake --build build -j

which places ``_perceptnet_cpp*.so`` inside ``perceptnet/ops``. If it is missing, every function
here transparently falls back to the NumPy reference implementation.
"""

from __future__ import annotations

import logging

import numpy as np

from perceptnet.data.preprocess import LetterboxInfo, preprocess_frame
from perceptnet.ops.boxes import iou_matrix as _iou_numpy
from perceptnet.ops.nms import batched_nms_numpy, nms_numpy

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on local build
    from perceptnet.ops import _perceptnet_cpp as _cpp

    HAS_CPP = True
    HAS_CPP_OPENCV = bool(getattr(_cpp, "has_opencv", False))
except ImportError:  # pragma: no cover
    _cpp = None
    HAS_CPP = False
    HAS_CPP_OPENCV = False


def _use_cpp(backend: str) -> bool:
    if backend == "cpp":
        if not HAS_CPP:
            raise RuntimeError("C++ backend requested but perceptnet.ops._perceptnet_cpp is not built")
        return True
    if backend == "numpy":
        return False
    if backend == "auto":
        return HAS_CPP
    raise ValueError(f"Unknown backend '{backend}'")


def iou_matrix(a: np.ndarray, b: np.ndarray, backend: str = "auto") -> np.ndarray:
    if _use_cpp(backend):
        return _cpp.iou_matrix(a, b)
    return _iou_numpy(a, b)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, backend: str = "auto") -> np.ndarray:
    if _use_cpp(backend):
        return _cpp.nms(boxes, scores, float(iou_threshold))
    return nms_numpy(boxes, scores, iou_threshold)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float, backend: str = "auto"
) -> np.ndarray:
    if _use_cpp(backend):
        return _cpp.batched_nms(boxes, scores, labels.astype(np.int32), float(iou_threshold))
    return batched_nms_numpy(boxes, scores, labels, iou_threshold)


def preprocess(
    frame_bgr: np.ndarray, input_hw: tuple[int, int], mean, std, backend: str = "auto"
) -> tuple[np.ndarray, LetterboxInfo]:
    """Letterbox + normalize one frame; uses the fused C++ kernel when available."""
    if _use_cpp(backend) and HAS_CPP_OPENCV:
        out, (scale, px, py, h, w) = _cpp.preprocess_bgr(
            np.ascontiguousarray(frame_bgr), int(input_hw[0]), int(input_hw[1]), list(mean), list(std)
        )
        return out, LetterboxInfo(scale, px, py, h, w)
    return preprocess_frame(frame_bgr, input_hw, mean, std)
