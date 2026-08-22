#!/usr/bin/env python3
"""Unified end-to-end experiment runner.

Per spec.md, this script:
1. Generates ground truth sliding windows (21, 42, 126, 252) from the test set.
2. For each DL model: HP search on a grid → final training with best HP →
   generate 252-length samples → trim to 21/42/126 → save checkpoint.
3. For each statistical model: single full-history training → generate 252-length
   samples → save checkpoint (including block bootstrap).
4. Uses the saved checkpoints to generate evaluation samples matching the test
   set size.
5. All output lands in STONKBENCH_OUTPUT_ROOT (default ~/outputs/).

Usage:
    python scripts/run_experiment.py --models quantgan timegrad ...
    python scripts/run_experiment.py --models-all
    python scripts/run_experiment.py --skip-hp             # skip HP search
    python scripts/run_experiment.py --smoke               # quick validation
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import get_output_root, get_run_id, set_run_id
from src.utils.preprocessed_data_utils import (
    load_dl_set,
    load_stats_set,
    resolve_dl_set_path,
    resolve_stats_set_path,
    sliding_window_2d,
)
from src.experiments.hp_configs import (
    DL_MODEL_KEYS,
    MODEL_HP_CONFIGS,
    HP_SEARCH_EPOCHS,
    FULL_TRAIN_EPOCHS,
    configs_for_model,
    full_train_metadata,
)
from src.experiments.core.registry import (
    ADAPTER_REGISTRY,
    STATISTICAL_MODEL_KEYS,
    VARIANT_TO_BASE,
    get_adapter,
)
from src.experiments.core.pipeline import run_model_experiment
from src.utils.device import device_to_str, get_device, log_device_context
import torch

ALL_DL = DL_MODEL_KEYS
ALL_STAT = list(STATISTICAL_MODEL_KEYS)
SEQ_LENGTHS = [21, 42, 126, 252]
GENERATION_LENGTH = 252
DEFAULT_SEED = 42


def _test_set_size(model_key: str) -> int:
    """Return the number of test windows available for generation."""
    if model_key in STATISTICAL_MODEL_KEYS:
        stats_set = load_stats_set(resolve_stats_set_path())
        test_len = stats_set["test_series"].shape[0]
    else:
        dl_set = load_dl_set(resolve_dl_set_path())
        test_len = dl_set["test_series"].shape[0]
    # Number of windows = test_len - GENERATION_LENGTH + 1 (stride=1)
    n_windows = max(1, test_len - GENERATION_LENGTH + 1)
    return n_windows


def _make_hp_summary_for_model(model_key: str) -> Dict[str, Any]:
    """Build synthetic HP summary using the single best config per model."""
    if model_key not in MODEL_HP_CONFIGS:
        return {"models": {model_key: {"best_config": {
            "config_id": "vendor_best", "is_vendor_default": True,
            "learning_rate": 1e-3, "batch_size": 64, "patience": 6,
            "mean_best_val_loss": 0.0, "extras": {},
        }}}}
    cfgs = MODEL_HP_CONFIGS[model_key]
    cfg = next((c for c in cfgs if getattr(c, "is_vendor_default", False)), cfgs[0])
    config = dataclasses.asdict(cfg)
    config["mean_best_val_loss"] = 0.0
    return {"models": {model_key: {"best_config": config}}}


def _hp_search_single_model(
    model_key: str, output_root: Path, run_id: str,
    dl_set: Dict[str, Any], device: str, smoke: bool = False,
) -> Dict[str, Any]:
    """Run HP search for a single DL model; return summary dict."""
    from src.experiments.hp_search import build_trial_specs, run_trial, aggregate_results

    hp_dir = output_root / "results" / run_id / "hp_search"
    trial_dir = hp_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    specs = build_trial_specs([model_key], seed=DEFAULT_SEED, smoke=smoke)
    results = []
    for spec in specs:
        trial_path = trial_dir / f"{spec.trial_id:05d}_{spec.label_with_smoke(smoke)}.json"
        if trial_path.exists():
            try:
                with trial_path.open() as f:
                    results.append(json.load(f))
                continue
            except (json.JSONDecodeError, OSError):
                pass
        print(f"  [HP] trial {spec.trial_id}: {spec.label_with_smoke(smoke)}")
        results.append(run_trial(spec, dl_set, device, hp_dir, smoke=smoke))

    summary = aggregate_results(results)
    summary_path = hp_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _generate_ground_truth(output_root: Path, run_id: str) -> None:
    """Generate ground truth sliding windows from the test set."""
    dl_set = load_dl_set(resolve_dl_set_path())
    test_series = dl_set["test_series"].float()
    gt_dir = output_root / "results" / run_id / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.preprocessed_data_utils import channel_norm_stats, denormalize_channels
    stats = channel_norm_stats(dl_set)

    for seq_len in SEQ_LENGTHS:
        windows = sliding_window_2d(test_series, seq_len, stride=1)
        if windows.shape[0] == 0:
            print(f"  [GT] seq{seq_len}: no windows")
            continue
        if stats is not None:
            windows = denormalize_channels(windows, *stats)
        gt_path = gt_dir / f"ground_truth_seq{seq_len}.pt"
        torch.save({"data": windows, "metadata": {"seq_len": seq_len, "n_windows": int(windows.shape[0])}}, gt_path)
        print(f"  [GT] seq{seq_len}: {windows.shape[0]} windows → {gt_path}")


def _training_metadata(
    model_key: str, hp_summary: Dict[str, Any], generation_length: int, smoke: bool,
) -> Dict[str, Any]:
    """Build training metadata from HP summary or defaults."""
    if model_key in STATISTICAL_MODEL_KEYS:
        return {"generation_length": generation_length}

    model_entry = hp_summary.get("models", {}).get(model_key)
    if not model_entry or not model_entry.get("best_config"):
        hp_summary = _make_hp_summary_for_model(model_key)
        model_entry = hp_summary["models"][model_key]

    meta = full_train_metadata(model_key, model_entry)
    if smoke:
        meta["max_epochs"] = 1
        meta["patience"] = 1
    meta["generation_length"] = generation_length
    return meta


def _run_dl_model(
    model_key: str, output_root: Path, run_id: str, device: str,
    hp_summary: Dict[str, Any], skip_hp: bool = False, smoke: bool = False,
) -> Dict[str, Any]:
    """Run HP search + final training + generation for a DL model."""
    print(f"\n{'='*60}")
    print(f"DL MODEL: {model_key}")
    print(f"{'='*60}")

    if not skip_hp and model_key in MODEL_HP_CONFIGS:
        print(f"  [HP] Running HP search...")
        hp_result = _hp_search_single_model(
            model_key, output_root, run_id,
            load_dl_set(resolve_dl_set_path()), device, smoke=smoke,
        )
        # Merge into cumulative summary
        hp_summary.setdefault("models", {}).update(hp_result.get("models", {}))
        best_cfg = hp_result.get("models", {}).get(model_key, {}).get("best_config", {})
        print(f"  [HP] Best: {best_cfg.get('config_id', '?')} lr={best_cfg.get('learning_rate', '?')}")
    elif model_key not in hp_summary.get("models", {}):
        hp_summary.setdefault("models", {}).update(
            _make_hp_summary_for_model(model_key).get("models", {})
        )

    meta = _training_metadata(model_key, hp_summary, GENERATION_LENGTH, smoke)
    num_samples = _test_set_size(model_key)
    run_model_experiment(
        model_key=model_key,
        generation_length=GENERATION_LENGTH,
        num_samples=num_samples,
        num_epochs=FULL_TRAIN_EPOCHS.get(model_key, 100) if not smoke else 1,
        seed=DEFAULT_SEED,
        device=device,
        output_root=output_root,
        training_metadata=meta,
        sanity_output_dir=output_root / "sanity" / run_id,
        seq_lengths=SEQ_LENGTHS,
        trim_from_max=True,
    )
    return hp_summary


def _run_stat_model(
    model_key: str, output_root: Path, run_id: str, device: str, smoke: bool = False,
) -> None:
    """Run single full-history training + generation for a statistical model."""
    print(f"\n{'='*60}")
    print(f"STAT MODEL: {model_key}")
    print(f"{'='*60}")

    # Bootstrap models generate natively at each seq_len (they accept
    # generation_length in generate() and support arbitrary lengths).
    # Non-bootstrap models (Merton, DEJD, GARCH) use trim_from_max
    # because they generate via Cholesky diffusion at the trained length.
    is_bootstrap = model_key in ("block_bootstrap", "stationary_block_bootstrap")
    num_samples = _test_set_size(model_key)
    run_model_experiment(
        model_key=model_key,
        generation_length=GENERATION_LENGTH,
        num_samples=num_samples,
        num_epochs=1,
        seed=DEFAULT_SEED,
        device=device,
        output_root=output_root,
        training_metadata={"generation_length": GENERATION_LENGTH},
        sanity_output_dir=output_root / "sanity" / run_id,
        seq_lengths=SEQ_LENGTHS,
        trim_from_max=not is_bootstrap,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified StonkBench experiment runner")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--models-all", action="store_true")
    p.add_argument("--skip-hp", action="store_true", help="Skip HP search")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else get_output_root()
    run_id = args.run_id or get_run_id()
    set_run_id(run_id)
    os.environ["STONKBENCH_RUN_ID"] = run_id

    device = device_to_str(get_device(args.device))
    print(log_device_context())
    print(f"Output root: {output_root}")
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")

    models = args.models or (ALL_DL + ALL_STAT if args.models_all else ALL_DL + ALL_STAT)
    t0 = time.perf_counter()

    # Step 1: Ground truth
    print("\n" + "="*60)
    print("STEP 1: Ground Truth Generation")
    print("="*60)
    _generate_ground_truth(output_root, run_id)

    # Step 2: DL models (include variant keys that resolve via VARIANT_TO_BASE)
    dl_models = [m for m in models if (m in ADAPTER_REGISTRY or m in VARIANT_TO_BASE) and m not in STATISTICAL_MODEL_KEYS]
    hp_summary: Dict[str, Any] = {"models": {}}
    if dl_models:
        print("\n" + "="*60)
        print("STEP 2: DL Models (HP search + final train + generate)")
        print("="*60)
        for model_key in dl_models:
            hp_summary = _run_dl_model(model_key, output_root, run_id, device, hp_summary, skip_hp=args.skip_hp, smoke=args.smoke)

    # Step 3: Statistical models
    stat_models = [m for m in models if m in STATISTICAL_MODEL_KEYS]
    if stat_models:
        print("\n" + "="*60)
        print("STEP 3: Statistical Models (full-history train + generate)")
        print("="*60)
        for model_key in stat_models:
            _run_stat_model(model_key, output_root, run_id, device, smoke=args.smoke)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
