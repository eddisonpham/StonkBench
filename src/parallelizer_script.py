import subprocess
from multiprocessing import Pool
import os

seq_lengths = [52]

def run_eval(seq_len):
    cmd = [
        "python", "src/unified_evaluator.py",
        "--seq_length", str(seq_len),
        "--num_samples", "1000",
        "--num_epochs", "10",
        "--results_dir", f"results/seq_A_{seq_len}"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    max_processes = min(len(seq_lengths), os.cpu_count())
    print(f"Running {max_processes} processes")
    with Pool(processes=max_processes) as pool:
        pool.map(run_eval, seq_lengths)
