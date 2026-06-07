"""Typed configuration for PerceptNet, loadable from YAML.

Every section is a frozen-free dataclass so values can be overridden from the CLI
(``--set train.lr=1e-3``) after loading.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # "coco" for a COCO-format annotation file, "synthetic" for generated shapes.
    kind: str = "synthetic"
    train_images: str = ""
    train_annotations: str = ""
    val_images: str = ""
    val_annotations: str = ""
    # Class names; for COCO data this is filled from the annotation file if empty.
    class_names: list[str] = field(default_factory=lambda: ["car", "pedestrian", "cyclist"])
    input_size: tuple[int, int] = (384, 640)  # (height, width) of the network input
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_objects: int = 64
    batch_size: int = 16
    num_workers: int = 4
    shuffle: bool = True
    # Augmentation
    hflip_prob: float = 0.5
    scale_jitter: tuple[float, float] = (0.75, 1.25)
    hsv_jitter: tuple[float, float, float] = (0.015, 0.5, 0.3)
    # Synthetic dataset size
    synthetic_train_size: int = 2000
    synthetic_val_size: int = 200

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass
class ModelConfig:
    backbone: str = "resnet18"  # resnet18 | resnet34
    neck_channels: tuple[int, ...] = (256, 128, 64)
    head_channels: int = 64
    output_stride: int = 4
    dtype: str = "float32"  # float32 | bfloat16 (compute dtype)
    heatmap_prior: float = 0.1  # initial foreground probability for heatmap bias


@dataclass
class LossConfig:
    heatmap_weight: float = 1.0
    size_weight: float = 0.1
    offset_weight: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 40
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 10.0
    # Fine-tuning controls
    pretrained_backbone: str = ""  # path to .npz/.msgpack backbone weights
    freeze_backbone_steps: int = 0  # backbone receives zero updates for this many steps
    backbone_lr_mult: float = 0.1
    freeze_backbone_bn: bool = False
    log_every: int = 50
    eval_every_epochs: int = 5
    output_dir: str = "runs/default"


@dataclass
class InferenceConfig:
    score_threshold: float = 0.3
    nms_iou_threshold: float = 0.5
    use_nms: bool = True  # CenterNet peaks are mostly NMS-free; NMS cleans residual duplicates
    top_k: int = 100
    batch_size: int = 1
    backend: str = "auto"  # auto | cpp | numpy  (for NMS post-processing)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(_jsonable(self.to_dict()), f, sort_keys=False)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _coerce(value: Any, template: Any) -> Any:
    """Coerce a YAML/CLI value to the type of the dataclass default."""
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(template, tuple):
        return tuple(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _update_section(section: Any, values: dict[str, Any]) -> None:
    valid = {f.name for f in dataclasses.fields(section)}
    for key, value in values.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {type(section).__name__}")
        setattr(section, key, _coerce(value, getattr(section, key)))


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML (optional) and apply ``section.key=value`` overrides."""
    cfg = Config()
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for section_name, values in raw.items():
            if not hasattr(cfg, section_name):
                raise KeyError(f"Unknown config section '{section_name}'")
            _update_section(getattr(cfg, section_name), values or {})
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        dotted, raw_value = item.split("=", 1)
        section_name, key = dotted.split(".", 1)
        value = yaml.safe_load(raw_value)
        _update_section(getattr(cfg, section_name), {key: value})
    return cfg
