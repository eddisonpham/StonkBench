"""Quick adapter pipeline smoke test (CPU or GPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiments.run_benchmark import main as run_benchmark_main
from src.utils.device import log_device_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test available benchmark adapters.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["quantgan", "gbm_adapter"],
        help="Subset of registry keys to exercise",
    )
    parser.add_argument("--generation_length", type=int, default=100)
    parser.add_argument("--output_root", type=str, default="src/experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(log_device_context())
    argv = [
        "smoke_test",
        "--smoke_test",
        "--generation_length",
        str(args.generation_length),
        "--models",
        *args.models,
        "--output_root",
        args.output_root,
    ]
    sys.argv = argv
    run_benchmark_main()


if __name__ == "__main__":
    main()
