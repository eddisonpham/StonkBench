import argparse
import subprocess
from pathlib import Path
from typing import List


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
        gen_cmd_param = [
            "python",
            "src/generation_scripts/generate_parametric_data.py",
            "--num_samples",
            str(args.num_samples),
        ]
        gen_cmd_param += ["--seq_lengths", *seq_lengths]

        gen_cmd_nonparam = [
            "python",
            "src/generation_scripts/generate_non_parametric_data.py",
            "--num_samples",
            str(args.num_samples),
            "--num_epochs",
            str(args.num_epochs),
        ]
        gen_cmd_nonparam += ["--seq_lengths", *seq_lengths]

        procs.append(run(gen_cmd_param))
        procs.append(run(gen_cmd_nonparam))

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
