from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput, StandardBatch
from src.experiments.core.io import append_jsonl, build_run_manifest, ensure_experiment_paths, write_json
from src.experiments.core.registry import STATISTICAL_MODEL_KEYS, get_adapter
from src.utils.device import device_to_str, get_device
from src.utils.artifact_utils import default_metadata, save_artifact
from src.utils.preprocessed_data_utils import (
    build_batch_from_dl_set,
    build_batch_from_stats_set,
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    load_stats_set,
    resolve_dl_set_path,
    resolve_stats_set_path,
    sliding_window_2d,
)


def _setup_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _prepare_standard_batch(model_key: str, generation_length: int) -> StandardBatch:
    if model_key in STATISTICAL_MODEL_KEYS:
        stats_set = load_stats_set(resolve_stats_set_path())
        return build_batch_from_stats_set(stats_set, generation_length=generation_length)
    dl_set = load_dl_set(resolve_dl_set_path())
    return build_batch_from_dl_set(dl_set, generation_length=generation_length)


def _preprocessing_metadata(model_key: str) -> Dict[str, Any]:
    dataset_kind = "statistical" if model_key in STATISTICAL_MODEL_KEYS else "deep_learning"
    return {
        "dataset_kind": dataset_kind,
        "dl_set_path": resolve_dl_set_path(),
        "stats_set_path": resolve_stats_set_path(),
    }


def run_model_experiment(
    model_key: str,
    generation_length: int,
    num_samples: int,
    num_epochs: int,
    seed: int,
    device: str,
    output_root: Path,
    training_metadata: Optional[Dict[str, Any]] = None,
    sanity_output_dir: Optional[Path] = None,
    sanity_price_assets: Tuple[str, str] = ("SPY", "AAPL"),
) -> Path:
    _setup_seed(seed)
    resolved_device = device_to_str(get_device(device))
    adapter = get_adapter(model_key)
    paths = ensure_experiment_paths(output_root, model_key)
    preprocessing = _preprocessing_metadata(model_key)
    batch = _prepare_standard_batch(model_key, generation_length)

    metadata: Dict[str, Any] = {"generation_length": generation_length}
    if training_metadata:
        metadata.update(training_metadata)

    fit_input = AdapterFitInput(
        batch=batch,
        sequence_length=batch.inferred_length or generation_length,
        num_epochs=num_epochs,
        device=resolved_device,
        seed=seed,
        metadata=metadata,
    )
    fit_info = adapter.fit(fit_input, checkpoints_dir=paths.checkpoints, logs_dir=paths.logs)
    generated = adapter.generate(num_samples=num_samples, generation_length=generation_length, seed=seed)

    # Map model outputs from normalized training space back to raw feature space.
    if model_key not in STATISTICAL_MODEL_KEYS:
        dl_set = load_dl_set(resolve_dl_set_path())
        stats = channel_norm_stats(dl_set)
        if stats is not None:
            mean, std = stats
            generated = AdapterGenerateOutput(
                data=denormalize_channels(generated.data, mean, std),
                checkpoints=generated.checkpoints,
                logs=generated.logs,
                extra_metadata=generated.extra_metadata,
            )

    metadata = default_metadata(
        model_name=model_key,
        model_type="statistical" if model_key in STATISTICAL_MODEL_KEYS else "deep_learning",
        sequence_length=generation_length,
        num_samples=num_samples,
        seed=seed,
        preprocessing_cfg=preprocessing,
        extra={
            "num_channels": int(generated.data.shape[-1]),
            "asset_columns": batch.asset_columns,
            "price_columns": batch.price_columns,
            "is_multivariate": True,
            "train_sequence_length": int(batch.inferred_length or generation_length),
            "model_checkpoint_manifest": [str(p) for p in generated.checkpoints],
            **fit_info,
            **generated.extra_metadata,
        },
    )

    artifact_path = paths.artifacts / f"{model_key}_seq_{generation_length}.pt"
    save_artifact(generated.data, metadata, artifact_path)

    manifest = build_run_manifest(
        model_name=model_key,
        adapter_name=adapter.__class__.__name__,
        cfg=preprocessing,
        checkpoint_paths=generated.checkpoints,
        extra={"artifact_path": str(artifact_path), "fit_info": fit_info},
    )
    write_json(paths.logs / "run_manifest.json", manifest)
    append_jsonl(paths.logs / "run.jsonl", {"event": "run_complete", "artifact": str(artifact_path)})

    if sanity_output_dir is not None and model_key not in STATISTICAL_MODEL_KEYS:
        from src.experiments.sanity_visualization import render_model_sanity

        sanity_paths = render_model_sanity(
            adapter=adapter,
            batch=batch,
            output_dir=sanity_output_dir / model_key,
            generation_length=generation_length,
            seed=seed,
            price_assets=sanity_price_assets,
        )
        write_json(
            sanity_output_dir / model_key / "manifest.json",
            {"model_key": model_key, "plots": [str(p) for p in sanity_paths]},
        )

    return artifact_path


def run_benchmark(
    model_keys: Iterable[str],
    generation_length: int,
    num_samples: int,
    num_epochs: int,
    seed: int,
    device: str,
    output_root: Path,
    training_metadata: Optional[Dict[str, Any]] = None,
    sanity_output_dir: Optional[Path] = None,
    sanity_price_assets: Tuple[str, str] = ("SPY", "AAPL"),
) -> List[Path]:
    artifacts = []
    for model_key in model_keys:
        artifacts.append(
            run_model_experiment(
                model_key=model_key,
                generation_length=generation_length,
                num_samples=num_samples,
                num_epochs=num_epochs,
                seed=seed,
                device=device,
                output_root=output_root,
                training_metadata=training_metadata,
                sanity_output_dir=sanity_output_dir,
                sanity_price_assets=sanity_price_assets,
            )
        )
    return artifacts

