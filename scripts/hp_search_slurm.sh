#!/bin/bash
#SBATCH --job-name=stonkbench-hp
#SBATCH --output=logs/hp_search_%A_%a.out
#SBATCH --error=logs/hp_search_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-431

set -euo pipefail

# 6 models x 24 HP configs x 3 seeds = 432 trials by default.
# Adjust --array to match `python -m src.experiments.hp_search --list_trials | tail -1`.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export STONKBENCH_DL_SET_PATH="${STONKBENCH_DL_SET_PATH:-data/preprocessed/dl_set.pt}"

mkdir -p logs results/hp_search

python -m src.experiments.hp_search \
  --device cuda \
  --output_dir results/hp_search \
  --trial_id "${SLURM_ARRAY_TASK_ID}"

# After all array tasks finish:
# python -m src.experiments.hp_search --aggregate_only --output_dir results/hp_search
