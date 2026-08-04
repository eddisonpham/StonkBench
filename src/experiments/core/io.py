from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.experiments.core.contracts import ExperimentPaths
from src.utils.artifact_utils import compute_preprocessing_hash
from src.utils.env import get_run_id as resolve_run_id  # canonical: see src/utils/env.py


# `resolve_run_id` is the imported re-export of `src.utils.env.get_run_id`
# declared at the top of this file. No local override; the import is canonical.
_ = None  # placeholder so neighbouring function bodies keep their indent


def results_root(output_root: Path, run_id: Optional[str] = None) -> Path:
    return output_root / "results" / resolve_run_id(run_id)


def ensure_experiment_paths(
    output_root: Path,
    model_name: str,
    run_id: Optional[str] = None,
) -> ExperimentPaths:
    rid = resolve_run_id(run_id)
    run_results = results_root(output_root, rid)
    paths = ExperimentPaths(
        root=output_root,
        model_root=output_root / "experiments" / rid / model_name,
        artifacts=run_results / model_name / "artifacts",
        checkpoints=output_root / "checkpoints" / rid / model_name,
        logs=output_root / "logs" / rid / model_name,
        metrics=run_results / model_name / "metrics",
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
        "run_id": resolve_run_id(),
        "config": cfg,
        "config_hash": compute_preprocessing_hash(cfg),
        "checkpoints": [str(p) for p in checkpoint_paths],
    }
    if extra:
        manifest.update(extra)
    return manifest


def experiment_paths_to_dict(paths: ExperimentPaths) -> Dict[str, str]:
    return {k: str(v) for k, v in asdict(paths).items()}
