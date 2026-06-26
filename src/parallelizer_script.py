import argparse
import subprocess
from pathlib import Path
from typing import List


DEFAULT_MODELS = [
    "quantgan",
    "timegan",
    "timegrad",
    "timevae",
    "unconditional_tsdiffusion",
    "vrnn",
    "gbm_adapter",
    "block_bootstrap",
]


def run(cmd: List[str]):
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel orchestration for generation and evaluation.")
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="*",
        default=[52, 60, 120, 180, 240, 300],
        help="Sequence lengths to process.",
    )
    parser.add_argument("--num_samples", type=int, default=1000, help="Samples per artifact.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Training epochs for deep models.")
    parser.add_argument(
        "--stage",
        choices=["generate", "evaluate", "all"],
        default="all",
        help="Stage to run.",
    )
    parser.add_argument("--max_procs", type=int, default=3, help="Maximum concurrent processes.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model keys passed to run_benchmark.",
    )
    # Evaluation arguments (passed through to unified_evaluator)
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=None,
        help="Directory containing generated artifacts.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to store evaluation outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seq_lengths = [str(s) for s in args.seq_lengths]

    procs = []

    if args.stage in ("generate", "all"):
        for seq_len in seq_lengths:
            gen_cmd = [
                "python",
                "-m",
                "src.experiments.run_benchmark",
                "--generation_length",
                seq_len,
                "--num_samples",
                str(args.num_samples),
                "--num_epochs",
                str(args.num_epochs),
                "--models",
                *args.models,
            ]
            procs.append(run(gen_cmd))

    if args.stage in ("evaluate", "all"):
        eval_cmd = [
            "python",
            "src/unified_evaluator.py",
            "--seq_lengths",
            *seq_lengths,
        ]
        if args.generated_dir:
            eval_cmd.extend(["--generated_dir", args.generated_dir])
        if args.results_dir:
            eval_cmd.extend(["--results_dir", args.results_dir])
        procs.append(run(eval_cmd))

    # Wait for all spawned processes
    for p in procs:
        p.wait()

    print("[DONE] All requested stages completed.")


if __name__ == "__main__":
    main()
