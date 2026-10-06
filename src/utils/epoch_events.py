"""Env-gated live training-curve events (JSONL) for the mini-experiment logger.

Completely inert unless ``STONKBENCH_MINI_EVENTS`` points at a writable file.
Used by DL adapters (and the vendored QuantGAN trainer) to stream per-epoch
train/val losses while a job is still running, so ``tail -f`` on that file
gives live training curves. Behavior-neutral: when the env var is unset this
is a single string check and a return — no I/O, no extra allocations.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


def emit_epoch(
    model_key: str,
    epoch: int,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    **extra: Any,
) -> None:
    """Append one training-curve event to ``$STONKBENCH_MINI_EVENTS``.

    Never raises: logging must not be able to break a training run.
    """
    path = os.environ.get("STONKBENCH_MINI_EVENTS", "").strip()
    if not path:
        return
    payload: dict[str, Any] = {
        "event": "epoch",
        "ts": time.time(),
        "model": str(model_key),
        "epoch": int(epoch),
    }
    if train_loss is not None:
        try:
            payload["train_loss"] = float(train_loss)
        except (TypeError, ValueError):
            payload["train_loss"] = str(train_loss)
    if val_loss is not None:
        try:
            payload["val_loss"] = float(val_loss)
        except (TypeError, ValueError):
            payload["val_loss"] = str(val_loss)
    for key, value in extra.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            payload[key] = value
        else:
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                payload[key] = str(value)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
