"""Msgpack checkpointing via ``flax.serialization`` (no extra checkpoint dependency)."""

from __future__ import annotations

import json
from pathlib import Path

from flax import serialization


def save_checkpoint(state, out_dir: str | Path, name: str = "last", extra: dict | None = None) -> Path:
    """Save the full train state (params, batch stats, optimizer, step)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.msgpack"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(serialization.to_bytes(state))
    tmp.replace(path)  # atomic on POSIX: never leaves a half-written checkpoint
    if extra is not None:
        (out / f"{name}.json").write_text(json.dumps(extra, indent=2))
    return path


def restore_checkpoint(state, path: str | Path):
    return serialization.from_bytes(state, Path(path).read_bytes())


def save_inference_variables(variables: dict, path: str | Path) -> Path:
    """Save only ``{'params', 'batch_stats'}`` — what the inference engine needs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.msgpack_serialize(serialization.to_state_dict(variables)))
    return path


def load_inference_variables(path: str | Path) -> dict:
    """Load variables saved by :func:`save_inference_variables` *or* a full train-state checkpoint."""
    raw = serialization.msgpack_restore(Path(path).read_bytes())
    if "params" not in raw:
        raise ValueError(f"{path} does not contain 'params'")
    return {"params": raw["params"], "batch_stats": raw.get("batch_stats", {}) or {}}
