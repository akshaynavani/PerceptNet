"""Loading pretrained backbone weights for fine-tuning.

Supported formats:

* ``.npz`` with flat keys ``params/backbone/<module>/.../<leaf>`` and
  ``batch_stats/backbone/<module>/.../<leaf>`` (produced by
  ``scripts/convert_torchvision_resnet.py`` from ImageNet ResNet safetensors).
* ``.msgpack`` containing ``{'params': ..., 'batch_stats': ...}`` from a previous PerceptNet run;
  only the ``backbone`` subtree is transferred, so detectors with a different class count can
  be fine-tuned from each other.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util

log = logging.getLogger(__name__)


def _load_flat(path: Path) -> dict[tuple[str, ...], np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {tuple(k.split("/")): data[k] for k in data.files}
    if path.suffix == ".msgpack":
        raw = serialization.msgpack_restore(path.read_bytes())
        flat = {}
        for collection in ("params", "batch_stats"):
            if collection in raw and "backbone" in raw[collection]:
                for k, v in traverse_util.flatten_dict(raw[collection]["backbone"]).items():
                    flat[(collection, "backbone", *k)] = np.asarray(v)
        return flat
    raise ValueError(f"Unsupported pretrained weight format: {path.suffix}")


def load_pretrained_backbone(variables: dict, path: str | Path, strict: bool = False) -> tuple[dict, dict]:
    """Overwrite backbone leaves in ``variables`` with pretrained values.

    Returns the updated variables and a report ``{'loaded': n, 'missing': [...], 'mismatched': [...]}``.
    """
    path = Path(path)
    pretrained = _load_flat(path)
    out = {}
    report = {"loaded": 0, "missing": [], "mismatched": [], "unused": []}
    used = set()
    for collection in ("params", "batch_stats"):
        flat = traverse_util.flatten_dict(variables.get(collection, {}))
        new_flat = {}
        for key, value in flat.items():
            full = (collection, *key)
            if key[0] != "backbone":
                new_flat[key] = value
                continue
            if full not in pretrained:
                report["missing"].append("/".join(full))
                new_flat[key] = value
                continue
            src = pretrained[full]
            used.add(full)
            if src.shape != value.shape:
                report["mismatched"].append(f"{'/'.join(full)}: {src.shape} vs {value.shape}")
                new_flat[key] = value
                continue
            new_flat[key] = jnp.asarray(src, dtype=value.dtype)
            report["loaded"] += 1
        out[collection] = traverse_util.unflatten_dict(new_flat)
    report["unused"] = ["/".join(k) for k in pretrained if k not in used]
    if strict and (report["missing"] or report["mismatched"]):
        raise ValueError(f"Pretrained backbone mismatch: {report}")
    log.info(
        "Loaded %d pretrained backbone tensors from %s (missing=%d, mismatched=%d, unused=%d)",
        report["loaded"], path, len(report["missing"]), len(report["mismatched"]), len(report["unused"]),
    )
    return out, report
