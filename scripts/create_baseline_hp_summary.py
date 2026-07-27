"""Create a baseline HP summary from vendor-default configs.

This lets final training run without first running a full HP search,
using the vendor-recommended learning rate, batch size, and patience
for every DL model. Statistical adapters ignore the HP summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Ensure repo root is on path so `import src...` works when the script is run
# directly from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.hp_configs import (  # noqa: E402
    DL_MODEL_KEYS,
    HP_SEARCH_EPOCHS,
    configs_for_model,
)


def create_baseline_summary() -> dict:
    summary = {"models": {}, "selection_metric": "best_val_loss"}
    for model_key in DL_MODEL_KEYS:
        configs = configs_for_model(model_key)
        vendor = next((c for c in configs if c.is_vendor_default), configs[0])
        entry = {
            "config_key": vendor.config_id,
            "config_id": vendor.config_id,
            "is_vendor_default": True,
            "max_epochs": HP_SEARCH_EPOCHS[model_key],
            "learning_rate": vendor.learning_rate,
            "batch_size": vendor.batch_size,
            "patience": vendor.patience,
            "mean_best_val_loss": 0.0,
            "std_best_val_loss": 0.0,
            "seeds": [42],
            "trials": [],
        }
        summary["models"][model_key] = {
            "best_config": entry,
            "ranked_configs": [entry],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a baseline HP summary from vendor-default configs."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_baseline_summary()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote baseline HP summary to {output_path}")


if __name__ == "__main__":
    main()
