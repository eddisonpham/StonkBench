from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.experiments.core.contracts import ExperimentPaths
from src.utils.artifact_utils import compute_preprocessing_hash


def ensure_experiment_paths(output_root: Path, model_name: str) -> ExperimentPaths:
    paths = ExperimentPaths(
        root=output_root,
        model_root=output_root / "experiments" / model_name,
        artifacts=output_root / "results" / model_name / "artifacts",
        checkpoints=output_root / "checkpoints" / model_name,
        logs=output_root / "logs" / model_name,
        metrics=output_root / "results" / model_name / "metrics",
    )
    for p in (paths.model_root, paths.artifacts, paths.checkpoints, paths.logs, paths.metrics):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def build_run_manifest(
    model_name: str,
    adapter_name: str,
    cfg: Dict[str, Any],
    checkpoint_paths: Iterable[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "adapter_name": adapter_name,
        "config": cfg,
        "config_hash": compute_preprocessing_hash(cfg),
        "checkpoints": [str(p) for p in checkpoint_paths],
    }
    if extra:
        manifest.update(extra)
    return manifest


def experiment_paths_to_dict(paths: ExperimentPaths) -> Dict[str, str]:
    return {k: str(v) for k, v in asdict(paths).items()}

