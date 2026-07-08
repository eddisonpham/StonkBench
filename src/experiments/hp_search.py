"""Hyperparameter search for DL models using validation loss only."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from src.experiments.core.registry import ADAPTER_REGISTRY
from src.experiments.core.contracts import AdapterFitInput
from src.experiments.core.registry import get_adapter
from src.experiments.hp_configs import (
    DL_MODEL_KEYS,
    HPConfig,
    HP_SEARCH_EPOCHS,
    configs_for_model,
)
from src.utils.device import device_to_str, get_device, log_device_context
from src.utils.preprocessed_data_utils import build_batch_from_dl_set, load_dl_set, resolve_dl_set_path

DEFAULT_SEED = 42


@dataclass(frozen=True)
class TrialSpec:
    model_key: str
    seed: int
    trial_id: int
    config: HPConfig

    def metadata(self, smoke: bool = False) -> Dict[str, Any]:
        max_epochs = 2 if smoke else HP_SEARCH_EPOCHS[self.model_key]
        return self.config.metadata(max_epochs=max_epochs)

    def label(self) -> str:
        meta = self.metadata()
        return (
            f"{self.model_key}_s{self.seed}_{self.config.config_id}_"
            f"e{meta['max_epochs']}_lr{meta['learning_rate']:g}_bs{meta['batch_size']}"
        )

    def label_with_smoke(self, smoke: bool) -> str:
        meta = self.metadata(smoke=smoke)
        return (
            f"{self.model_key}_s{self.seed}_{self.config.config_id}_"
            f"e{meta['max_epochs']}_lr{meta['learning_rate']:g}_bs{meta['batch_size']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DL HP search with validation-loss selection.")
    parser.add_argument("--dl_set_path", type=str, default=None, help="Override dl_set.pt path")
    parser.add_argument("--output_dir", type=str, default="/home/epham/StonkBench/output/results/hp_search")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available else cpu)",
    )
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Subset of DL model keys")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trial_id", type=int, default=None, help="Run a single flattened trial id")
    parser.add_argument("--list_trials", action="store_true", help="Print trial schedule and exit")
    parser.add_argument("--smoke", action="store_true", help="Tiny grid for quick validation")
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Rebuild summary.json from existing trials/ without training",
    )
    return parser.parse_args()


def build_trial_specs(
    model_keys: Sequence[str],
    seed: int,
    smoke: bool = False,
) -> List[TrialSpec]:
    specs: List[TrialSpec] = []
    trial_id = 0
    for model_key in model_keys:
        if smoke:
            configs = [configs_for_model(model_key)[0]]
        else:
            configs = configs_for_model(model_key)
        for config in configs:
            specs.append(
                TrialSpec(
                    model_key=model_key,
                    seed=int(seed),
                    trial_id=trial_id,
                    config=config,
                )
            )
            trial_id += 1
    return specs


def run_trial(
    spec: TrialSpec,
    dl_set: Dict[str, Any],
    device: str,
    output_dir: Path,
    smoke: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(spec.seed)
    batch = build_batch_from_dl_set(dl_set, generation_length=int(dl_set["window_size"]))
    adapter = get_adapter(spec.model_key)

    with tempfile.TemporaryDirectory(prefix=f"hp_{spec.label()}_") as tmp:
        tmp_path = Path(tmp)
        checkpoints_dir = tmp_path / "checkpoints"
        logs_dir = tmp_path / "logs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        fit_input = AdapterFitInput(
            batch=batch,
            sequence_length=int(batch.inferred_length or dl_set["window_size"]),
            num_epochs=1,
            device=device,
            seed=spec.seed,
            metadata=spec.metadata(smoke=smoke),
        )
        fit_info = adapter.fit(fit_input, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)

    meta = spec.metadata(smoke=smoke)
    result = {
        "trial_id": spec.trial_id,
        "model_key": spec.model_key,
        "seed": spec.seed,
        "label": spec.label_with_smoke(smoke),
        "config_id": spec.config.config_id,
        "is_vendor_default": spec.config.is_vendor_default,
        **meta,
        **fit_info,
    }
    trial_path = output_dir / "trials" / f"{spec.trial_id:05d}_{spec.label()}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    with trial_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def aggregate_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in results:
        model_key = row["model_key"]
        config_key = row.get("config_id") or (
            f"e{row['max_epochs']}_lr{row['learning_rate']:g}_bs{row['batch_size']}_p{row['patience']}"
        )
        grouped.setdefault(model_key, {}).setdefault(config_key, []).append(row)

    summary: Dict[str, Any] = {"models": {}, "selection_metric": "best_val_loss"}
    for model_key, configs in grouped.items():
        ranked = []
        for config_key, rows in configs.items():
            val_losses = [float(r["best_val_loss"]) for r in rows if "best_val_loss" in r]
            if not val_losses:
                continue
            ranked.append(
                {
                    "config_key": config_key,
                    "config_id": rows[0].get("config_id", config_key),
                    "is_vendor_default": bool(rows[0].get("is_vendor_default", False)),
                    "max_epochs": rows[0]["max_epochs"],
                    "learning_rate": rows[0]["learning_rate"],
                    "batch_size": rows[0]["batch_size"],
                    "patience": rows[0]["patience"],
                    "mean_best_val_loss": float(statistics.mean(val_losses)),
                    "std_best_val_loss": float(statistics.pstdev(val_losses)) if len(val_losses) > 1 else 0.0,
                    "seeds": [int(r["seed"]) for r in rows],
                    "trials": rows,
                }
            )
        ranked.sort(key=lambda item: item["mean_best_val_loss"])
        summary["models"][model_key] = {
            "best_config": ranked[0] if ranked else None,
            "ranked_configs": ranked,
        }
    return summary


def main() -> None:
    args = parse_args()
    model_keys = args.models or DL_MODEL_KEYS
    unknown = [m for m in model_keys if m not in ADAPTER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    specs = build_trial_specs(model_keys=model_keys, seed=args.seed, smoke=args.smoke)
    if args.list_trials:
        for spec in specs:
            print(f"{spec.trial_id}\t{spec.label()}")
        print(f"total_trials={len(specs)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        trial_dir = output_dir / "trials"
        if not trial_dir.exists():
            raise FileNotFoundError(f"No trials found under {trial_dir}")
        results = []
        for path in sorted(trial_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
        summary = aggregate_results(results)
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregated {len(results)} trials -> {summary_path}")
        return

    if args.dl_set_path:
        import os

        os.environ["STONKBENCH_DL_SET_PATH"] = args.dl_set_path
    dl_set = load_dl_set(resolve_dl_set_path())
    if "valid_windows" not in dl_set or dl_set["valid_windows"].shape[0] == 0:
        raise ValueError(
            "dl_set.pt has no validation windows. Re-run preprocessing with --val_ratio, e.g.\n"
            "  python -m src.data_preprocessing"
        )

    device = device_to_str(get_device(args.device))
    print(log_device_context())

    if args.trial_id is not None:
        if args.trial_id < 0 or args.trial_id >= len(specs):
            raise ValueError(f"trial_id must be in [0, {len(specs) - 1}]")
        results = [run_trial(specs[args.trial_id], dl_set, device, output_dir, smoke=args.smoke)]
    else:
        results = []
        for spec in specs:
            print(f"Running trial {spec.trial_id}/{len(specs)-1}: {spec.label_with_smoke(args.smoke)}")
            results.append(run_trial(spec, dl_set, device, output_dir, smoke=args.smoke))

    summary = aggregate_results(results)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
