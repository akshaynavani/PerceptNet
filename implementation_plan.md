# PerceptNet — Implementation Plan

**Autonomous Perception Pipeline.** A high-performance 2D object detection pipeline in Python and
C++: a fine-tuned convolutional backbone written in **JAX/Flax**, trained with **Optax**, and executed
as an **XLA-compiled** (jit + ahead-of-time lowered) program for GPU-accelerated inference. Image frames
are preprocessed with **OpenCV**; hot post-processing kernels (NMS, letterbox) have a **C++** implementation
exposed through pybind11; numeric operations are benchmarked across NumPy, JAX and C++; and model accuracy
is evaluated across score and IoU thresholds.

## Key technical decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | JAX + Flax (linen) + Optax | Required stack; functional params make freezing/fine-tuning explicit |
| Compilation | `jax.jit`, plus AOT `jit(...).lower(...).compile()` per input shape | Fixed input resolution → one compiled XLA executable, no retrace at inference time |
| Detector | CenterNet-style anchor-free detector (heatmap + size + offset heads) | No anchors, NMS-light, decode is a pure array program that compiles cleanly under XLA |
| Backbone | ResNet-18/34 in Flax with stride-4 FPN-style upsampling neck | Standard, small enough to fine-tune on a single GPU |
| Fine-tuning | Load pretrained backbone weights from `.npz`/msgpack; freeze for N warmup steps via `optax.multi_transform`; lower backbone LR afterwards | "Fine-tuned backbone" story without needing PyTorch at runtime |
| Data | COCO-format JSON (works for KITTI/BDD100K exported to COCO) + a synthetic shapes dataset for smoke runs | One loader, real or synthetic data |
| Preprocessing | OpenCV letterbox resize, BGR→RGB, normalize; NumPy target encoding | Matches deployment-time preprocessing exactly |
| C++ | `cpp/` CMake project: NMS + IoU + letterbox, pybind11 module `_perceptnet_cpp`, standalone benchmark binary | Python falls back to NumPy if the extension is not built |
| Checkpoints | `flax.serialization` msgpack | No extra dependency |
| Evaluation | COCO-style AP@[.5:.95], AP50, AP75, per-class AP, P/R/F1 sweep across score thresholds | "Evaluated accuracy across detection thresholds" |

## Phases

### Phase 1 — Scaffolding, configuration and data pipeline
Deliverables:
- Packaging (`pyproject.toml`), package skeleton, dataclass config with YAML loading, default config.
- OpenCV preprocessing: letterbox resize, normalization, inverse box mapping back to original frame.
- COCO-format dataset reader, synthetic shapes dataset generator, augmentations (h-flip, HSV jitter, scale jitter).
- Batched data loader producing fixed-shape NumPy batches.

Files: `pyproject.toml`, `configs/*.yaml`, `perceptnet/__init__.py`, `perceptnet/config.py`,
`perceptnet/data/{preprocess,dataset,synthetic,augment,loader}.py`

### Phase 2 — Model, targets, losses and decoding (JAX/Flax)
Deliverables:
- Box utilities and IoU (NumPy + JAX), NumPy NMS reference.
- Flax ResNet backbone (BasicBlock, ResNet-18/34), upsampling neck, CenterNet heads, `PerceptNet` detector module.
- Gaussian heatmap target encoding, penalty-reduced focal loss, L1 size/offset losses.
- Jittable decode: heatmap max-pool peak extraction, top-K, box assembly.

Files: `perceptnet/ops/{boxes,nms}.py`, `perceptnet/model/{backbone,neck,heads,detector,targets,losses,decode}.py`

### Phase 3 — Training and fine-tuning
Deliverables:
- Train state with BatchNorm statistics, Optax schedule (warmup + cosine), backbone freezing / LR multiplier via `optax.multi_transform`.
- Pretrained backbone loading from `.npz` (with a torchvision-ResNet → Flax key mapping script), msgpack checkpointing.
- Jitted train/eval steps, training loop with logging, `perceptnet-train` CLI.

Files: `perceptnet/train/{state,optim,checkpoint,pretrained,trainer}.py`, `perceptnet/cli.py`,
`scripts/convert_torchvision_resnet.py`

### Phase 4 — XLA inference engine and C++ kernels
Deliverables:
- `InferenceEngine`: AOT-compiled XLA executable for a fixed batch/resolution, device placement, warmup, HLO/StableHLO dump, end-to-end `detect(frames)`.
- Image / video / webcam inference CLI with OpenCV visualization.
- C++: IoU, greedy + class-aware NMS, OpenCV letterbox; pybind11 bindings; CMake build; Python wrapper with NumPy fallback; standalone C++ benchmark.

Files: `perceptnet/inference/{engine,visualize}.py`, `perceptnet/ops/cpp_ext.py`,
`cpp/CMakeLists.txt`, `cpp/include/perceptnet/*.hpp`, `cpp/src/*.cpp`, `cpp/bench/bench_ops.cpp`

### Phase 5 — Evaluation, benchmarking and tests
Deliverables:
- Detection metrics: per-class AP (101-point interpolation), AP@[.5:.95], AP50/AP75, P/R/F1 sweep across score thresholds, JSON report.
- Benchmark suite: NumPy vs JAX (jit, CPU/GPU) vs C++ for IoU matrix, NMS, letterbox, conv/matmul; end-to-end latency/FPS jit vs eager across batch sizes.
- `perceptnet-eval`, `perceptnet-bench` CLIs; pytest suite for preprocessing, targets, decode, NMS parity, metrics, model shapes.

Files: `perceptnet/eval/{metrics,evaluate}.py`, `perceptnet/bench/{ops_bench,latency_bench}.py`, `tests/*.py`

Followed by `.gitignore` and `README.md` (architecture diagram, run and test instructions).
