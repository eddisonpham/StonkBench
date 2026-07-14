"""Stage-event JSONL emitter for StonkBench run telemetry.

Single source of truth for in-process shape traces, train metrics, progress,
and error halts. Mirror to stdout (``EVENT:<TYPE>:<json>``) so log-tail watchers
can attach to a run without disk access. JSONL is also written under
``log_dir/<model_key>/events.jsonl`` for permanent replay.

Torch-free at module-import time. Tensor ``shape`` / ``dtype`` / ``device`` are
extracted via duck-typed attribute access so this module loads without torch.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _shape_of(x: Any) -> List[int]:
    if hasattr(x, "shape"):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def _dtype_of(x: Any) -> str:
    return str(getattr(x, "dtype", "unknown"))


def _device_of(x: Any) -> str:
    return str(getattr(x, "device", "cpu"))


def _run_id_or_default(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    # Defer import so that core/events.py can be loaded without torch.
    from src.utils.env import get_run_id

    return get_run_id()


def emit(
    model_key: str,
    event_type: str,
    payload: Dict[str, Any],
    log_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> None:
    """Core write. Mirrors to stdout AND optionally appends to ``events.jsonl``."""
    evt = {
        "ts": _now_iso(),
        "run_id": _run_id_or_default(run_id),
        "model_key": model_key,
        "event": event_type,
        **payload,
    }
    line = json.dumps(evt, default=str)
    sys.stdout.write(f"EVENT:{event_type.upper()}:{line}\n")
    sys.stdout.flush()
    if log_dir is not None:
        model_log = Path(log_dir) / model_key
        model_log.mkdir(parents=True, exist_ok=True)
        path = model_log / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def init_config(
    model_key: str,
    hparams: Dict[str, Any],
    dataset_hash: str,
    gpu: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> None:
    emit(model_key, "init_config", {
        "hparams": dict(hparams),
        "dataset_hash": str(dataset_hash),
        "gpu": str(gpu) if gpu is not None else "cpu",
    }, log_dir=log_dir)


def shape_trace(
    model_key: str,
    stage: str,
    tensor_name: str,
    tensor: Any,
    extra: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None,
) -> None:
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "tensor_name": str(tensor_name),
        "shape": _shape_of(tensor),
        "dtype": _dtype_of(tensor),
        "device": _device_of(tensor),
    }
    if extra:
        payload["extra"] = dict(extra)
    emit(model_key, "shape_trace", payload, log_dir=log_dir)


def reshape_event(
    model_key: str,
    stage: str,
    original_shape: Any,
    new_shape: Any,
    reason: str,
    log_dir: Optional[Path] = None,
) -> None:
    emit(model_key, "reshape_event", {
        "stage": str(stage),
        "original_shape": _shape_of(original_shape),
        "new_shape": _shape_of(new_shape),
        "reason": str(reason),
    }, log_dir=log_dir)


def train_metric(
    model_key: str,
    epoch: int,
    metric: str,
    value: float,
    log_dir: Optional[Path] = None,
) -> None:
    emit(model_key, "train_metric", {
        "epoch": int(epoch),
        "metric": str(metric),
        "value": float(value),
    }, log_dir=log_dir)


def squeue_progress(
    model_key: str,
    epoch: int,
    total_epochs: int,
    log_dir: Optional[Path] = None,
) -> None:
    emit(model_key, "squeue_progress", {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
    }, log_dir=log_dir)


def error_halt(
    model_key: str,
    error_msg: str,
    traceback_str: str = "",
    log_dir: Optional[Path] = None,
) -> None:
    emit(model_key, "error_halt", {
        "error": str(error_msg),
        "traceback": str(traceback_str)[:4000],
    }, log_dir=log_dir)
