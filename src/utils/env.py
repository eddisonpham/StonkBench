"""Centralized environment accessors and defaults for StonkBench.

Replaces ad-hoc `os.environ` reads scattered across modules. Honors the
single resolution chain: explicit kwarg > env var > built-in default.

All other modules should import from here (NOT os.environ) so behavior is
predictable and testable.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# device.py imports torch; defer to first use of get_pipeline_device_* so env-only
# callers (tests, SLURM env-detect scripts, etc.) do not require torch.


# Canonical output root lives at <repo>/outputs/. STONKBENCH_OUTPUT_ROOT env
# still wins over this default; explicit CLI flags win over both.
DEFAULT_OUTPUT_ROOT_NAME = "outputs"
DEFAULT_DL_SET_RELATIVE = "data/preprocessed/dl_set.pt"
DEFAULT_STATS_SET_RELATIVE = "data/preprocessed/statsmodel_set.pt"

_BOOL_TRUE = {"1", "true", "yes", "on"}


def project_root() -> Path:
    """Repo root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(explicit: Optional[Path], env_name: str, default: Path) -> Path:
    """Path resolution chain shared by all path helpers."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default


def get_output_root(explicit: Optional[Path] = None) -> Path:
    """Output root directory. Priority: explicit > ``STONKBENCH_OUTPUT_ROOT`` > ``<repo>/outputs``."""
    return _resolve_path(
        explicit,
        "STONKBENCH_OUTPUT_ROOT",
        project_root() / DEFAULT_OUTPUT_ROOT_NAME,
    )


def get_dl_set_path(explicit: Optional[Path] = None) -> Path:
    """Path to preprocessed DL set. Priority: explicit > ``STONKBENCH_DL_SET_PATH`` > default."""
    return _resolve_path(
        explicit,
        "STONKBENCH_DL_SET_PATH",
        project_root() / DEFAULT_DL_SET_RELATIVE,
    )


def get_stats_set_path(explicit: Optional[Path] = None) -> Path:
    """Path to preprocessed statistical set."""
    return _resolve_path(
        explicit,
        "STONKBENCH_STATS_SET_PATH",
        project_root() / DEFAULT_STATS_SET_RELATIVE,
    )


def get_run_id(explicit: Optional[str] = None) -> str:
    """Dated run id (``YYYY-MM-DD`` by default).

    Override via CLI flag or ``STONKBENCH_RUN_ID``. To disambiguate multiple
    runs in the same day, set the env var to ``YYYY-MM-DD-HHMM`` manually.
    """
    if explicit:
        return explicit
    env = os.environ.get("STONKBENCH_RUN_ID", "").strip()
    if env:
        return env
    return datetime.now().strftime("%Y-%m-%d")


def set_run_id(run_id: str) -> None:
    """Pin run id into the environment so child processes inherit it."""
    os.environ["STONKBENCH_RUN_ID"] = run_id


def is_smoke(explicit: Optional[bool] = None) -> bool:
    """True when a smoke budget should be used (smaller batches/epochs)."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("STONKBENCH_SMOKE", "0").strip().lower()
    return raw in _BOOL_TRUE


def get_local_jobs(default: int = 1) -> int:
    """Process-pool concurrency ceiling for the local Python DAG driver.

    Override with ``STONKBENCH_LOCAL_JOBS=N``. Clamped to ``[1, cpu_count]`` so a
    runaway env var on a laptop cannot swamp the host. Reads ``os.environ`` and
    ``os.cpu_count()`` on every call — do not cache (env vars can change).
    """
    raw = os.environ.get("STONKBENCH_LOCAL_JOBS", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    upper = os.cpu_count() or 1
    return max(1, min(value, upper))


def get_visible_cuda_indices() -> list[int]:
    """Resolve CUDA indices honoring ``CUDA_VISIBLE_DEVICES`` / ``STONKBENCH_CUDA_VISIBLE_DEVICES``.

    Returns an ordered list of cuda indices as the orchestrator should use them
    (round-robin across this list). Empty list means: no CUDA available.
    """
    override = os.environ.get("STONKBENCH_CUDA_VISIBLE_DEVICES", "").strip()
    if override:
        return [int(x) for x in re.split(r"[,\s]+", override) if x != ""]
    env_default = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_default:
        return [int(x) for x in re.split(r"[,\s]+", env_default) if x != ""]
    return [0]


def _resolve_device_for_slot(device_spec: str):
    """Lazy torch importer (device.py pulls in torch; env.py stays torch-free at import)."""
    from src.utils.device import get_device

    return get_device(device_spec)


def get_pipeline_device_for_slot(slot_index: int = 0):
    """Round-robin device getter for the local DAG driver (Phase 1 locked decision)."""
    indices = get_visible_cuda_indices()
    if not indices:
        return _resolve_device_for_slot("cpu")
    target = f"cuda:{indices[slot_index % len(indices)]}"
    return _resolve_device_for_slot(target)


# Alias for clarity at call sites.
get_pipeline_device = get_pipeline_device_for_slot


def apply_run_id_override(args_run_id: str) -> str:
    """CLI ``--run_id`` value takes precedence; write it back so subprocesses inherit it."""
    chosen = args_run_id or get_run_id()
    set_run_id(chosen)
    return chosen
