#!/bin/bash
#SBATCH --job-name=simv2-eval
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/simeval_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/simeval_%j.err

# Unified evaluation suite over the 28 simulation_v2 artifacts (artifact-only,
# skip_regenerate default = evaluate the stored tensors as-is). Writes the
# standard layout: <results_dir>/complete_evaluation.json + seq_L/<model>/
# (QQ plots, distribution, tsne, per-asset figures, taxonomy metrics).
#
# Run as a Slurm job (not a login-shell background process) so it survives
# client session cleanup. Serial by design: every artifact writes into the
# shared results_dir tree and parallel writers would race on metrics.json
# (see scripts/vm/run_parallel.sh notes).
#
# Submit from trig-login01:
#   command sbatch --export=ALL scripts/mini_exp/eval_simv2.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

python -m src.unified_evaluator \
  --generated_dir "${OUTPUT_ROOT}/results/simulation_v2" \
  --results_dir "${OUTPUT_ROOT}/../mini_exp/simulation_v2/evaluation"

echo "=== SIMV2 EVAL done rc=$? $(date -Is) ==="
