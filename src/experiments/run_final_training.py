"""Final full training run using HP-search winners and post-training sanity plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.core.io import resolve_run_id, results_root
from src.experiments.core.pipeline import run_model_experiment
from src.experiments.core.registry import ADAPTER_REGISTRY, STATISTICAL_MODEL_KEYS
from src.experiments.hp_configs import DL_MODEL_KEYS as HP_DL_MODEL_KEYS
from src.experiments.hp_configs import full_train_metadata
from src.utils.device import device_to_str, get_device, log_device_context
from src.utils.env import get_output_root

DEFAULT_OUTPUT_ROOT = get_output_root()
ALL_MODEL_KEYS = sorted(ADAPTER_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final models from HP-search winners.")
    parser.add_argument("--hp_summary", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_id", type=str, default="", help="Dated results subfolder (STONKBENCH_RUN_ID)")
    # Default sequence length = 252 (≈1 trading year of daily bars). This
    # only sets the per-model `--generation_length` flag; the actual train
    # window L comes from `dl_set["window_size"]` set during preprocessing
    # (`python -m src.data_preprocessing --window_size 252`). If those two
    # values disagree, the adapter will silently stitch short windows up
    # to 252 on every generate() call. Keep them in sync.
    parser.add_argument("--generation_length", type=int, default=252)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_KEYS)
    parser.add_argument("--smoke", action="store_true", help="Tiny sample/epoch budget for validation")
    parser.add_argument("--skip_sanity", action="store_true")
    return parser.parse_args()


def _load_hp_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"HP summary not found at {path}. Run HP search and aggregate first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _training_metadata(
    model_key: str,
    hp_summary: Dict[str, Any],
    generation_length: int,
    smoke: bool,
) -> Dict[str, Any]:
    if model_key in STATISTICAL_MODEL_KEYS:
        return {"generation_length": generation_length}

    if model_key not in HP_DL_MODEL_KEYS:
        raise ValueError(f"No HP configuration for model '{model_key}'")

    model_entry = hp_summary.get("models", {}).get(model_key)
    if not model_entry or not model_entry.get("best_config"):
        raise ValueError(f"HP summary has no best_config for '{model_key}'")

    metadata = full_train_metadata(model_key, model_entry)
    if smoke:
        metadata["max_epochs"] = 1
        metadata["patience"] = 1
    metadata["generation_length"] = generation_length
    return metadata


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.run_id:
        os.environ["STONKBENCH_RUN_ID"] = args.run_id
    run_id = resolve_run_id(args.run_id or None)
    os.environ["STONKBENCH_RUN_ID"] = run_id
    run_results = results_root(output_root, run_id)
    hp_summary_path = Path(args.hp_summary) if args.hp_summary else run_results / "hp_search" / "summary.json"
    hp_summary = _load_hp_summary(hp_summary_path)

    device = device_to_str(get_device(args.device))
    print(log_device_context())
    print(f"Run id: {run_id}")
    print(f"HP summary: {hp_summary_path}")
    print(f"Output root: {output_root}")
    print(f"Results dir: {run_results}")

    num_samples = 16 if args.smoke else args.num_samples
    num_epochs = 1 if args.smoke else 1  # adapters honor metadata max_epochs for DL models
    sanity_dir = None if args.skip_sanity else output_root / "sanity" / run_id

    artifacts: List[Path] = []
    for model_key in args.models:
        if model_key not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}")

        training_metadata = _training_metadata(
            model_key=model_key,
            hp_summary=hp_summary,
            generation_length=args.generation_length,
            smoke=args.smoke,
        )
        print(f"Training {model_key} with metadata={training_metadata}")
        artifact = run_model_experiment(
            model_key=model_key,
            generation_length=args.generation_length,
            num_samples=num_samples,
            num_epochs=num_epochs,
            seed=args.seed,
            device=device,
            output_root=output_root,
            training_metadata=training_metadata,
            sanity_output_dir=sanity_dir,
        )
        artifacts.append(artifact)

    print("Saved artifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")
    if sanity_dir is not None:
        print(f"Sanity plots under: {sanity_dir}")


if __name__ == "__main__":
    main()
