import subprocess
import os

seq_lengths = [52,60,120,180,240,300]

def run_eval(seq_len):
    cmd = [
        "python", "src/unified_evaluator.py",
        "--seq_length", str(seq_len),
        "--num_samples", "1000",
        "--num_epochs", "10",
        "--results_dir", f"results/seq_{seq_len}"
    ]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    # Number of parallel processes set to 3, or less if not enough CPUs or sequence lengths
    max_processes = min(6, len(seq_lengths), os.cpu_count())
    print(f"Running up to {max_processes} processes in parallel")
    processes = []
    seq_idx = 0

    # Main loop: launch processes up to max_processes in parallel
    while seq_idx < len(seq_lengths) or processes:
        # Launch new processes if slots available
        while seq_idx < len(seq_lengths) and len(processes) < max_processes:
            seq_len = seq_lengths[seq_idx]
            print(f"Launching process for seq_len={seq_len}")
            p = run_eval(seq_len)
            processes.append(p)
            seq_idx += 1
        # Clean finished processes from the list
        for p in processes[:]:
            if p.poll() is not None:
                processes.remove(p)
        # Avoid tight loop
        if len(processes) == max_processes or (seq_idx == len(seq_lengths) and processes):
            # Wait a moment before next check
            try:
                # Wait for one to finish, but keep checking processes in a loop
                processes[0].wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    print("All processes finished.")
