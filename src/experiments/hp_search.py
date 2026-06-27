"""Hyperparameter search for DL models using validation loss only."""

from __future__ import annotations

import argparse
import itertools
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
from src.utils.preprocessed_data_utils import build_batch_from_dl_set, load_dl_set, resolve_dl_set_path

DL_MODEL_KEYS = [
    "quantgan",
    "timegan",
    "timegrad",
    "timevae",
    "unconditional_tsdiffusion",
    "vrnn",
]

DEFAULT_SEEDS = [7, 11, 42]

HP_GRID: Dict[str, List[Any]] = {
    "max_epochs": [10, 20, 30, 40],
    "learning_rate": [1e-4, 1e-3, 3e-3],
    "batch_size": [32, 64],
    "patience": [12],
}


@dataclass(frozen=True)
class TrialSpec:
    model_key: str
    seed: int
    trial_id: int
    max_epochs: int
    learning_rate: float
    batch_size: int
    patience: int

    def metadata(self) -> Dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "use_calibration": False,
        }

    def label(self) -> str:
        return (
            f"{self.model_key}_s{self.seed}_e{self.max_epochs}_"
            f"lr{self.learning_rate:g}_bs{self.batch_size}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DL HP search with validation-loss selection.")
    parser.add_argument("--dl_set_path", type=str, default=None, help="Override dl_set.pt path")
    parser.add_argument("--output_dir", type=str, default="results/hp_search")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Subset of DL model keys")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trial_id", type=int, default=None, help="Run a single flattened trial id")
    parser.add_argument("--list_trials", action="store_true", help="Print trial schedule and exit")
    parser.add_argument("--smoke", action="store_true", help="Tiny grid for quick validation")
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Rebuild summary.json from existing trials/ without training",
    )
    return parser.parse_args()


def _grid_values(smoke: bool) -> Dict[str, List[Any]]:
    if not smoke:
        return HP_GRID
    return {
        "max_epochs": [2],
        "learning_rate": [1e-3],
        "batch_size": [32],
        "patience": [12],
    }


def build_trial_specs(
    model_keys: Sequence[str],
    seeds: Sequence[int],
    smoke: bool = False,
) -> List[TrialSpec]:
    grid = _grid_values(smoke)
    combos = list(
        itertools.product(
            grid["max_epochs"],
            grid["learning_rate"],
            grid["batch_size"],
            grid["patience"],
        )
    )
    specs: List[TrialSpec] = []
    trial_id = 0
    for model_key in model_keys:
        for seed in seeds:
            for max_epochs, learning_rate, batch_size, patience in combos:
                specs.append(
                    TrialSpec(
                        model_key=model_key,
                        seed=int(seed),
                        trial_id=trial_id,
                        max_epochs=int(max_epochs),
                        learning_rate=float(learning_rate),
                        batch_size=int(batch_size),
                        patience=int(patience),
                    )
                )
                trial_id += 1
    return specs


def run_trial(spec: TrialSpec, dl_set: Dict[str, Any], device: str, output_dir: Path) -> Dict[str, Any]:
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
            metadata=spec.metadata(),
        )
        fit_info = adapter.fit(fit_input, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)

    result = {
        "trial_id": spec.trial_id,
        "model_key": spec.model_key,
        "seed": spec.seed,
        "label": spec.label(),
        **spec.metadata(),
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
        config_key = (
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

    specs = build_trial_specs(model_keys=model_keys, seeds=args.seeds, smoke=args.smoke)
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

    if args.trial_id is not None:
        if args.trial_id < 0 or args.trial_id >= len(specs):
            raise ValueError(f"trial_id must be in [0, {len(specs) - 1}]")
        results = [run_trial(specs[args.trial_id], dl_set, args.device, output_dir)]
    else:
        results = []
        for spec in specs:
            print(f"Running trial {spec.trial_id}/{len(specs)-1}: {spec.label()}")
            results.append(run_trial(spec, dl_set, args.device, output_dir))

    summary = aggregate_results(results)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
