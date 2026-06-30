"""XLA-compiled inference engine.

The forward pass *and* CenterNet decoding (sigmoid, 3x3 peak max-pool, top-K, box assembly)
are fused into a single jitted function, then lowered and compiled ahead of time for a fixed
``(batch, H, W, 3)`` input. At runtime there is no tracing and no Python in the hot path:
one compiled XLA executable per engine, launched once per batch.

Post-processing that is cheap and data-dependent (score thresholding, residual NMS, mapping
back to original-frame coordinates) runs on the host, using the C++ kernels when built.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from perceptnet.config import Config
from perceptnet.data.preprocess import LetterboxInfo, boxes_to_original
from perceptnet.model.decode import decode
from perceptnet.model.detector import build_model
from perceptnet.ops import cpp_ext
from perceptnet.train.checkpoint import load_inference_variables

log = logging.getLogger(__name__)


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy in original-frame pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) int32

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class Timings:
    preprocess_ms: list[float] = field(default_factory=list)
    device_ms: list[float] = field(default_factory=list)
    postprocess_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        out = {}
        for name in ("preprocess_ms", "device_ms", "postprocess_ms"):
            vals = getattr(self, name)
            if vals:
                out[f"{name}_p50"] = float(np.percentile(vals, 50))
                out[f"{name}_p95"] = float(np.percentile(vals, 95))
        return out


def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    info: LetterboxInfo,
    icfg,
    score_threshold: float | None = None,
) -> Detections:
    """Host-side post-processing for one image's top-K decoded detections.

    Score threshold -> optional class-aware NMS (C++ when built) -> map to original-frame pixels.
    """
    thr = icfg.score_threshold if score_threshold is None else score_threshold
    keep = scores > thr
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if icfg.use_nms and len(scores) > 1:
        idx = cpp_ext.batched_nms(boxes, scores, classes, icfg.nms_iou_threshold, backend=icfg.backend)
        boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
    boxes = boxes_to_original(boxes, info)
    return Detections(boxes.astype(np.float32), scores.astype(np.float32), classes.astype(np.int32))


class InferenceEngine:
    def __init__(
        self,
        cfg: Config,
        variables: dict,
        batch_size: int | None = None,
        device: jax.Device | None = None,
        aot: bool = True,
    ) -> None:
        self.cfg = cfg
        self.input_hw = tuple(cfg.data.input_size)
        self.batch_size = batch_size or cfg.inference.batch_size
        self.stride = cfg.model.output_stride
        self.top_k = cfg.inference.top_k
        self.device = device or jax.devices()[0]
        self.model = build_model(cfg)
        self.variables = jax.device_put(variables, self.device)
        self.timings = Timings()

        model, top_k, stride = self.model, self.top_k, self.stride

        def forward(variables, images):
            outputs = model.apply(variables, images, train=False)
            return decode(outputs, top_k=top_k, stride=stride)

        self._jitted = jax.jit(forward)
        self._input_spec = jax.ShapeDtypeStruct((self.batch_size, *self.input_hw, 3), jnp.float32)
        self.compile_time_s = 0.0
        self._compiled = None
        if aot:
            t0 = time.perf_counter()
            self._lowered = self._jitted.lower(self.variables, self._input_spec)
            self._compiled = self._lowered.compile()
            self.compile_time_s = time.perf_counter() - t0
            log.info(
                "AOT-compiled PerceptNet for input %s on %s in %.2fs",
                self._input_spec.shape, self.device, self.compile_time_s,
            )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(cls, cfg: Config, checkpoint: str | Path, **kwargs) -> InferenceEngine:
        return cls(cfg, load_inference_variables(checkpoint), **kwargs)

    # ------------------------------------------------------------------ introspection
    def stablehlo_text(self) -> str:
        """The StableHLO module JAX hands to XLA (before XLA optimizations)."""
        return self._jitted.lower(self.variables, self._input_spec).as_text()

    def optimized_hlo_text(self) -> str:
        """The post-optimization HLO for the compiled executable (fusions, layouts...)."""
        if self._compiled is None:
            raise RuntimeError("Engine was built with aot=False")
        return self._compiled.as_text()

    def cost_analysis(self) -> dict:
        """XLA's static cost estimate (flops, bytes accessed) when the backend provides one."""
        if self._compiled is None:
            return {}
        try:
            cost = self._compiled.cost_analysis()
        except Exception:  # pragma: no cover - backend specific
            return {}
        if isinstance(cost, list):
            cost = cost[0] if cost else {}
        return {k: float(v) for k, v in (cost or {}).items() if k in ("flops", "bytes accessed", "transcendentals")}

    def dump_hlo(self, out_dir: str | Path) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / "perceptnet.stablehlo.mlir"]
        paths[0].write_text(self.stablehlo_text())
        if self._compiled is not None:
            paths.append(out / "perceptnet.optimized.hlo.txt")
            paths[1].write_text(self.optimized_hlo_text())
        return paths

    # ------------------------------------------------------------------ execution
    def infer_batch(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the compiled graph on a preprocessed (B, H, W, 3) batch.

        Returns raw top-K detections in network-input pixels: boxes (B,K,4), scores, classes.
        """
        if images.shape != self._input_spec.shape:
            raise ValueError(f"Expected batch of shape {self._input_spec.shape}, got {images.shape}")
        x = jax.device_put(images.astype(np.float32, copy=False), self.device)
        fn = self._compiled if self._compiled is not None else self._jitted
        boxes, scores, classes = fn(self.variables, x)
        boxes.block_until_ready()
        return np.asarray(boxes), np.asarray(scores), np.asarray(classes)

    def warmup(self, iterations: int = 3) -> None:
        dummy = np.zeros(self._input_spec.shape, np.float32)
        for _ in range(iterations):
            self.infer_batch(dummy)

    def postprocess(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, info: LetterboxInfo,
        score_threshold: float | None = None,
    ) -> Detections:
        return postprocess_detections(boxes, scores, classes, info, self.cfg.inference, score_threshold)

    def detect(self, frames_bgr: list[np.ndarray], score_threshold: float | None = None) -> list[Detections]:
        """End-to-end detection on raw BGR frames of any size."""
        dcfg = self.cfg.data
        results: list[Detections] = []
        for start in range(0, len(frames_bgr), self.batch_size):
            chunk = frames_bgr[start : start + self.batch_size]
            t0 = time.perf_counter()
            batch = np.zeros(self._input_spec.shape, np.float32)
            infos = []
            for i, frame in enumerate(chunk):
                batch[i], info = cpp_ext.preprocess(
                    frame, self.input_hw, dcfg.mean, dcfg.std, backend=self.cfg.inference.backend
                )
                infos.append(info)
            t1 = time.perf_counter()
            boxes, scores, classes = self.infer_batch(batch)
            t2 = time.perf_counter()
            for i, info in enumerate(infos):
                results.append(self.postprocess(boxes[i], scores[i], classes[i], info, score_threshold))
            t3 = time.perf_counter()
            self.timings.preprocess_ms.append((t1 - t0) * 1e3)
            self.timings.device_ms.append((t2 - t1) * 1e3)
            self.timings.postprocess_ms.append((t3 - t2) * 1e3)
        return results
