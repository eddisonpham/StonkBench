"""Unified PyTorch device resolution for local CPU and Slurm GPU runs."""

from __future__ import annotations

import os
from typing import Union

import torch

DeviceLike = Union[str, torch.device]


def get_device(requested: DeviceLike | None = None) -> torch.device:
    """Pick CUDA when requested and available; otherwise CPU."""
    if isinstance(requested, torch.device):
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    preferred = requested or os.environ.get("STONKBENCH_DEVICE", "cuda")
    if str(preferred).startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred if ":" in str(preferred) else "cuda")
    return torch.device("cpu")


def device_to_str(device: DeviceLike) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def resolve_device(device: DeviceLike | None = None) -> torch.device:
    """Alias used by DL adapters — same policy as :func:`get_device`."""
    return get_device(device)


def log_device_context(prefix: str = "device") -> str:
    """Return a one-line summary for job logs."""
    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"{prefix}={device} ({name})"
    return f"{prefix}={device} (CUDA unavailable, using CPU)"
