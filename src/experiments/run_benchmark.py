from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.core.pipeline import run_model_experiment
from src.utils.device import device_to_str, get_device, log_device_context
from src.utils.preprocessed_data_utils import DL_SET_PATH, STATS_SET_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified multivariate adapter benchmark runner.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["quantgan", "timegan", "timegrad", "timevae", "unconditional_tsdiffusion", "vrnn"],
    )
    parser.add_argument("--generation_length", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available else cpu; STONKBENCH_DEVICE env overrides)",
    )
    parser.add_argument("--output_root", type=str, default="/home/epham/StonkBench/output")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        num_samples = min(args.num_samples, 16)
        num_epochs = min(args.num_epochs, 1)
    else:
        num_samples = args.num_samples
        num_epochs = args.num_epochs

    device = device_to_str(get_device(args.device))
    print(log_device_context())
    print(f"DL set: {DL_SET_PATH}")
    print(f"Stats set: {STATS_SET_PATH}")

    artifacts = []
    for model_key in args.models:
        artifacts.append(
            run_model_experiment(
                model_key=model_key,
                generation_length=args.generation_length,
                num_samples=num_samples,
                num_epochs=num_epochs,
                seed=args.seed,
                device=device,
                output_root=Path(args.output_root),
            )
        )
    print("Saved artifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")


if __name__ == "__main__":
    main()
