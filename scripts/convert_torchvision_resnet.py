#!/usr/bin/env python
"""Convert ImageNet ResNet-18/34 weights (torchvision/timm key layout, safetensors) to PerceptNet .npz.

No PyTorch required: weights are read with ``safetensors.numpy``.

    pip install safetensors huggingface_hub
    python scripts/convert_torchvision_resnet.py --arch resnet34 --out weights/resnet34_imagenet.npz

By default the safetensors file is fetched from the Hugging Face Hub (timm/<arch>.a1_in1k).
Pass ``--safetensors path/to/model.safetensors`` to convert a local file instead.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_BN_LEAVES = {"weight": ("params", "scale"), "bias": ("params", "bias"),
              "running_mean": ("batch_stats", "mean"), "running_var": ("batch_stats", "var")}


def map_key(key: str) -> tuple[str, str] | None:
    """Map a torch-style key to ``(collection, flax_path)`` or None if it should be skipped."""
    if key.startswith("fc.") or key.endswith("num_batches_tracked"):
        return None
    parts = key.split(".")
    leaf = parts[-1]
    module = parts[:-1]
    # layerX.Y.<...> -> layerX_Y/<...>
    if re.fullmatch(r"layer\d", module[0]):
        module = [f"{module[0]}_{module[1]}"] + module[2:]
    # downsample.0 -> downsample_conv, downsample.1 -> downsample_bn
    if len(module) >= 3 and module[-2] == "downsample":
        module = module[:-2] + ["downsample_conv" if module[-1] == "0" else "downsample_bn"]
    name = module[-1]
    is_conv = name.startswith("conv") or name == "downsample_conv"
    if is_conv:
        if leaf != "weight":
            return None
        return "params", "/".join(["backbone", *module, "kernel"])
    if leaf not in _BN_LEAVES:
        return None
    collection, flax_leaf = _BN_LEAVES[leaf]
    return collection, "/".join(["backbone", *module, flax_leaf])


def convert(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for key, value in state.items():
        mapped = map_key(key)
        if mapped is None:
            continue
        collection, path = mapped
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 4:  # OIHW -> HWIO
            value = value.transpose(2, 3, 1, 0)
        out[f"{collection}/{path}"] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--safetensors", default="", help="local model.safetensors (skips download)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors.numpy import load_file

    path = args.safetensors
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(f"timm/{args.arch}.a1_in1k", "model.safetensors")
    converted = convert(load_file(path))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **converted)
    print(f"wrote {len(converted)} tensors to {args.out}")


if __name__ == "__main__":
    main()
