"""Train state carrying BatchNorm running statistics alongside params and optimizer state."""

from __future__ import annotations

from typing import Any

from flax.training import train_state


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def create_train_state(model, variables, tx) -> TrainState:
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables.get("batch_stats", {}),
        tx=tx,
    )
