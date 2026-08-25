#!/bin/bash
#SBATCH --job-name=sb-merge-plot
#SBATCH --account=def-yqhuang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/merge_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/merge_%j.err

# Merge per-task evaluation results into complete_evaluation.json, then generate plots.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

GENERATED_DIR="${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}"
RESULTS_DIR="${GENERATED_DIR}/evaluation"
PLOTS_DIR="${GENERATED_DIR}/plots"

echo "=== MERGE: Collecting per-task results $(date -Is) ==="

# Merge all per-model-per-seq metrics.json into complete_evaluation.json
python -c "
import json
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
all_results = {}

for metrics_file in sorted(results_dir.glob('seq_*/*/metrics.json')):
    try:
        with open(metrics_file) as f:
            data = json.load(f)
    except Exception:
        continue
    model = data.get('model_name', '')
    seq = data.get('evaluated_at_length', data.get('sequence_length', 0))
    key = f'{model}_seq{seq}'
    all_results[key] = data

summary_path = results_dir / 'complete_evaluation.json'
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'Merged {len(all_results)} results -> {summary_path}')
"

echo "=== PLOTS: Generating evaluation plots $(date -Is) ==="

python -m src.plot_statistics.evaluation_plots \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${PLOTS_DIR}"

echo "=== DONE $(date -Is) ==="
echo "Results: ${RESULTS_DIR}"
echo "Plots:   ${PLOTS_DIR}"
