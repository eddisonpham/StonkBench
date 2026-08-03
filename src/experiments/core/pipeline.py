from __future__ import annotations

import time
import traceback as _traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from src.experiments.core import events
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput, StandardBatch
from src.experiments.core.io import append_jsonl, build_run_manifest, ensure_experiment_paths, write_json
from src.experiments.core.registry import STATISTICAL_MODEL_KEYS, get_adapter
from src.utils.device import device_to_str, get_device
from src.utils.artifact_utils import compute_preprocessing_hash, default_metadata, save_artifact, stitch_sequences
from src.utils.artifact_archival import archive_existing
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
    seq_lengths: Optional[List[int]] = None,
    trim_from_max: bool = False,
) -> List[Path]:
    _setup_seed(seed)
    resolved_device = device_to_str(get_device(device))
    adapter = get_adapter(model_key)
    paths = ensure_experiment_paths(output_root, model_key)
    preprocessing = _preprocessing_metadata(model_key)
    batch = _prepare_standard_batch(model_key, generation_length)
    if generation_length > batch.test.shape[0]:
        raise ValueError(
            f"{model_key}: generation_length={generation_length} exceeds test series length "
            f"{batch.test.shape[0]}. Choose a shorter horizon or a longer dataset."
        )

    metadata: Dict[str, Any] = {
        "generation_length": generation_length,
        # Per-(model, seq) checkpoint filename (`{model_key}_seq{L}_final.pt`)
        # inside each adapter requires the registry key, not the class attr.
        # Thread it in via metadata so adapters stay free of registry queries.
        "model_key": model_key,
    }
    if training_metadata:
        metadata.update(training_metadata)

    # Telemetry: emit init_config + shape_trace at pipeline.entry so log-tail
    # watchers and JSONL replay get one canonical "training is about to start"
    # event per model run.
    hparams: Dict[str, Any] = {"model_key": model_key, **metadata, "device": resolved_device}
    try:
        events.init_config(
            model_key=model_key,
            hparams=hparams,
            dataset_hash=compute_preprocessing_hash(preprocessing),
            gpu=resolved_device,
            log_dir=output_root / "logs",
        )
        events.shape_trace(
            model_key=model_key,
            stage="adapter.fit.start",
            tensor_name="train_windows" if batch.train_windows is not None else "train_series",
            tensor=batch.train_windows if batch.train_windows is not None else batch.train,
            extra={"sequence_length": int(batch.inferred_length or generation_length)},
            log_dir=output_root / "logs",
        )
    except Exception:  # noqa: BLE001 — telemetry must never crash the run.
        print(f"[WARN] Failed to emit pipeline telemetry for {model_key}; continuing.")

    fit_input = AdapterFitInput(
        batch=batch,
        sequence_length=batch.inferred_length or generation_length,
        num_epochs=num_epochs,
        device=resolved_device,
        seed=seed,
        metadata=metadata,
    )
    # Move any existing items for the same (model, seq) out of the live
    # outputs tree into ``outputs/_legacy/{utc-ts}_{model}_seq{L}/`` so this
    # retrain writes fresh. Idempotent: returns 0 (no-op) if nothing existed.
    # The active ``outputs/results/{run_id}/{model}/`` and
    # ``outputs/sanity/{run_id}/{model}/`` for this combo are then empty,
    # which matches the user's standing instruction that "the outputs folder
    # is completely clean and we dont have any confusing dated reruns."
    archive_existing(model_key, generation_length, output_root)
    # Wall-clock: capture per-model fit+generate duration. Emitted as
    # `run_start` + `run_end` events so log replay and run.jsonl summaries
    # can compute time-taken per model without scanning stdout.
    t_start = time.perf_counter()
    events.run_start(
        model_key=model_key,
        hparams=hparams,
        log_dir=output_root / "logs",
    )
    # Determine which seq lengths to generate.  If caller passes
    # ``seq_lengths`` we train once and generate at every requested length;
    # otherwise we fall back to the legacy single-length behaviour.
    _gen_lengths = seq_lengths if seq_lengths else [generation_length]
    max_gen_len = max(_gen_lengths)

    try:
        fit_info = adapter.fit(fit_input, checkpoints_dir=paths.checkpoints, logs_dir=paths.logs)
        elapsed_sec = float(time.perf_counter() - t_start)
        events.run_end(
            model_key=model_key,
            elapsed_sec=elapsed_sec,
            fit_summary={"best_val_loss": float(fit_info.get("best_val_loss", float("nan"))),
                          "best_epoch": int(fit_info.get("best_epoch", 0)),
                          "stopped_early": bool(fit_info.get("stopped_early", False)),
                          "num_channels": int(fit_info.get("num_channels", 1))},
            log_dir=output_root / "logs",
        )
        fit_info = dict(fit_info)
        fit_info["wall_clock_seconds"] = elapsed_sec
    except Exception as exc:  # noqa: BLE001
        elapsed_sec = float(time.perf_counter() - t_start)
        try:
            events.run_end(model_key=model_key, elapsed_sec=elapsed_sec, error=str(exc), log_dir=output_root / "logs")
            events.error_halt(model_key=model_key, error_msg=str(exc),
                              traceback_str=_traceback.format_exc(), log_dir=output_root / "logs")
        except Exception:  # noqa: BLE001
            print(f"[WARN] Failed to emit error_halt for {model_key}; tearing down anyway.")
        raise

    # Denormalization stats (loaded once for DL models).
    dl_norm_stats = None
    if model_key not in STATISTICAL_MODEL_KEYS:
        dl_set = load_dl_set(resolve_dl_set_path())
        dl_norm_stats = channel_norm_stats(dl_set)

    # --- Generate, denormalize, save, GT, and sanity for each seq_len ---
    # Trim mode: honor `--trim_from_max` by generating a single base tensor at
    # max(_gen_lengths) and slicing/stitching to each requested seq_len. This
    # makes the adapter contract honest (we don't lie about
    # `supports_arbitrary_generation`) and ensures shorter windows are exact
    # prefixes of the same underlying 252 tensor (deterministic, identical RNG).
    # Without this flag, the legacy per-iter generate() path is used (each
    # seq_len gets its own native draw).
    #
    # METHODOLOGICAL CAVEAT: when --trim_from_max is set, seq21 / seq42 /
    # seq126 are prefixes of one 252-tensor, NOT independent draws. Downstream
    # metrics that treat per-seq_len samples as independent (KS-tests, moment
    # divergence, etc.) will be correlated across seq_len. Document this when
    # reporting results.
    base_gen: Optional[AdapterGenerateOutput] = None
    if trim_from_max:
        base_gen = adapter.generate(num_samples=num_samples, generation_length=max_gen_len, seed=seed)

    artifacts: List[Path] = []
    for seq_len in _gen_lengths:
        # Generate
        if trim_from_max:
            if base_gen is None:
                # Should be unreachable: trim_from_max pre-generates above.
                # If it fires, the caller passed an empty seq_lengths list —
                # which is a caller-side bug, not a pipeline bug.
                raise RuntimeError(
                    "trim_from_max=True requires a non-empty _gen_lengths; "
                    "check caller's seq_lengths / generation_length."
                )
            if base_gen.data.shape[1] == seq_len:
                # Aliasing the cached tensor is safe: the denorm block below
                # wraps it in a fresh AdapterGenerateOutput with a NEW data
                # tensor, so base_gen.data is never mutated in place.
                generated = base_gen
            else:
                # stitch_sequences handles target<base as a pure slice; for
                # target>base it would tile (not what we want — the caller
                # validates seq_len <= max(_gen_lengths) up front).
                if seq_len > base_gen.data.shape[1]:
                    raise ValueError(
                        f"trim_from_max requires seq_len <= base length "
                        f"({seq_len} > {base_gen.data.shape[1]})"
                    )
                generated = AdapterGenerateOutput(
                    data=stitch_sequences(base_gen.data, seq_len, seed=seed),
                    checkpoints=base_gen.checkpoints, logs=base_gen.logs,
                    extra_metadata=base_gen.extra_metadata,
                )
        else:
            if adapter.supports_arbitrary_generation:
                native_length = seq_len
            else:
                native_length = int(batch.inferred_length or seq_len)
            generated = adapter.generate(num_samples=num_samples, generation_length=native_length, seed=seed)
            if generated.data.shape[1] != seq_len:
                generated = AdapterGenerateOutput(
                    data=stitch_sequences(generated.data, seq_len, seed=seed),
                    checkpoints=generated.checkpoints, logs=generated.logs,
                    extra_metadata=generated.extra_metadata,
                )
        # Denormalize
        if dl_norm_stats is not None:
            mean, std = dl_norm_stats
            data = denormalize_channels(generated.data, mean, std).detach().cpu()
            generated = AdapterGenerateOutput(data=data, checkpoints=generated.checkpoints,
                                             logs=generated.logs, extra_metadata=generated.extra_metadata)
        else:
            generated = AdapterGenerateOutput(data=generated.data.detach().cpu(),
                                             checkpoints=generated.checkpoints,
                                             logs=generated.logs, extra_metadata=generated.extra_metadata)

        metadata = default_metadata(
            model_name=model_key,
            model_type="statistical" if model_key in STATISTICAL_MODEL_KEYS else "deep_learning",
            sequence_length=seq_len, num_samples=num_samples, seed=seed,
            preprocessing_cfg=preprocessing,
            extra={
                "num_channels": int(generated.data.shape[-1]),
                "asset_columns": batch.asset_columns, "price_columns": batch.price_columns,
                "is_multivariate": True,
                "train_sequence_length": int(batch.inferred_length or max_gen_len),
                "model_checkpoint_manifest": [str(p) for p in generated.checkpoints],
                **fit_info, **generated.extra_metadata,
            },
        )
        artifact_path = paths.artifacts / f"{model_key}_seq{seq_len}.pt"
        save_artifact(generated.data, metadata, artifact_path)
        artifacts.append(artifact_path)

        # Ground truth — fresh sliding windows per seq_len.
        # For DL models, denormalize test_series to raw space so GT matches
        # the denormalized artifacts.
        gt_path = output_root / "ground_truth" / f"ground_truth_seq{seq_len}.pt"
        test_series = batch.test.float()
        if dl_norm_stats is not None:
            mean_gt, std_gt = dl_norm_stats
            test_series = denormalize_channels(test_series, mean_gt, std_gt)
        gt_windows = sliding_window_2d(test_series, seq_len, stride=1)
        if gt_windows.shape[0] > 0:
            gt_meta = default_metadata(
                model_name="ground_truth", model_type="ground_truth",
                sequence_length=seq_len, num_samples=int(gt_windows.shape[0]),
                seed=seed, preprocessing_cfg=preprocessing,
                extra={"num_channels": int(gt_windows.shape[-1]),
                       "asset_columns": batch.asset_columns, "price_columns": batch.price_columns,
                       "is_multivariate": True},
            )
            save_artifact(gt_windows, gt_meta, gt_path)

        # Sanity plots
        if sanity_output_dir is not None and model_key not in STATISTICAL_MODEL_KEYS:
            try:
                from src.experiments.sanity_visualization import render_model_sanity
                sanity_model_dir = sanity_output_dir / model_key / f"seq{seq_len}"
                sanity_model_dir.mkdir(parents=True, exist_ok=True)
                sanity_paths = render_model_sanity(
                    adapter=adapter, batch=batch, output_dir=sanity_model_dir,
                    generation_length=seq_len, seed=seed,
                )
                print(f"  [sanity] {model_key} seq{seq_len}: {len(sanity_paths)} plots")
            except Exception as e:
                print(f"  [sanity-error] {model_key} seq{seq_len}: {e}")

    manifest = build_run_manifest(
        model_name=model_key, adapter_name=adapter.__class__.__name__,
        cfg=preprocessing, checkpoint_paths=getattr(adapter, "checkpoints", []),
        extra={"artifacts": [str(p) for p in artifacts], "fit_info": fit_info},
    )
    write_json(paths.logs / "run_manifest.json", manifest)
    append_jsonl(paths.logs / "run.jsonl", {"event": "run_complete", "artifacts": [str(p) for p in artifacts]})

    return artifacts


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
            )
        )
    return artifacts

