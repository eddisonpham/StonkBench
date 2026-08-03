#!/bin/bash
#SBATCH --job-name=sb-train-dl
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00   # Trillium `compute` partition max is 1 day; fits worst-case (timegrad 100ep ~8h on H100).
#SBATCH --array=0-7%8
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.err

# One DL model per GPU. 8 canonical DL models (the ones in MODEL_HP_CONFIGS)
# are trained in parallel (~1-3h each; 4d wall clock is generous headroom for
# cond_sig_wgan / kalman_vae / diffusion first-epoch ramp-ups).
#
# METHODOLOGICAL CAVEAT — trim mode:
#   --trim_from_max produces a SINGLE 252-length tensor per model, and the
#   shorter seq_lengths (21/42/126) are exact prefixes of that draw. This is
#   what the script does by default because you asked for "generate 252 and
#   trim". The PRACTICAL consequence: per-seq_len samples are NOT independent
#   draws. If downstream evaluation (unified_evaluator KS-tests, moment
#   divergence, etc.) treats them as independent samples, the metrics WILL
#   be artificially correlated across seq_len. Document this in any
#   write-up that references these artifacts.
#
# Trim semantics:
#   - GENERATION_LENGTH (default 252) is the native train/generate horizon
#     and is the length of the underlying draw.
#   - SEQ_LENGTHS (default "21 42 126") are produced by slicing the cached
#     252-tensor via pipeline.run_model_experiment's `trim_from_max=True`
#     branch. Shorter seq_lengths are deterministic prefixes.
#   - To get independent draws per seq_len (the legacy behavior), replace
#     --trim_from_max with nothing (or pass it as a CLI flag to NOT use it).
#     This file currently hard-enables it for the per-iter trim semantics
#     requested in the task.
#
# Hyperparameter source:
#   - This script intentionally does NOT supply --hp_summary. The runner
#     (`run_final_training.main`) falls back to `_make_smoke_hp_summary`,
#     which stamps the in-repo `MODEL_HP_CONFIGS["vendor_best"]` configs
#     for each requested model. This is the SINGLE SOURCE OF TRUTH for
#     HP. To inject an externally-tuned single HP (e.g. one tuned on a
#     non-Trillium VM), EITHER:
#       (a) edit `learning_rate` / `batch_size` / `patience` / `extras` for
#           the affected model in `src/experiments/hp_configs.py`, or
#       (b) drop a `summary.json` matching the format produced by
#           `_make_smoke_hp_summary` (see `src/experiments/hp_search.py`'s
#           `aggregate_results` for the exact schema) and set
#           `--hp_summary <path>` on the python invocation below.
#   If neither is done, the script silently uses the commit-time bake-in
#   `vendor_best` HPs. This is the expected behavior for the user's
#   external single-HP scenario (validated in the prior verification
#   round).
#
# Data wiring:
#   - `python -m src.data_preprocessing --window_size 252` produces the
#     canonical dl_set.pt at data/preprocessed/. This repo currently ships
#     the 4 window-specific files dl_set_W{21,42,126,252}.pt, so we override
#     STONKBENCH_DL_SET_PATH below to point at the 252 window file.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

# Override the default dl_set path. common.sh points to `dl_set.pt`, but
# this repo carries window-specific files (dl_set_W{21,42,126,252}.pt) so
# training lands on the correct window slice. Hard-set (no `:-` fallback):
# the prior `:-` form did not fire because common.sh already exports
# STONKBENCH_DL_SET_PATH=.../dl_set.pt, so the override was a no-op and the
# script crashed on the existence check below.
#
# If your account has the canonical dl_set.pt produced by
# `python -m src.data_preprocessing`, comment out the next line.
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set_W252.pt"
if [[ ! -f "${STONKBENCH_DL_SET_PATH}" ]]; then
  echo "[FATAL] DL set missing at ${STONKBENCH_DL_SET_PATH}" >&2
  echo "        Re-run: python -m src.data_preprocessing --window_size 252" >&2
  exit 2
fi

# 8 DL models — index 0..7 maps 1:1 to SLURM_ARRAY_TASK_ID.
# When STONKBENCH_FIX_VARIANTS=1 is set in the sbatch --export=, the variant
# set from src/experiments/core/registry.VARIANT_TO_BASE is used instead.
if [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "2" ]]; then
  # Wave 2: GAN collapse recovery (2026-07-30) — 6 variants for 3 collapsed GANs.
  # Submit with: sbatch --array=0-5%3 --export=ALL,STONKBENCH_FIX_VARIANTS=2 scripts/slurm/final_training.sh
  DL_MODELS=(
    quantgan_clip05_d3
    quantgan_clip10_n10
    cond_sig_wgan_st3k_h100
    cond_sig_wgan_p10_s2
  )
elif [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "4" ]]; then
  # Wave 3 + 4 architecture-fix batch (2026-07-30):
  # This slot retunes one variant at a time as new adapter paths ship.
  # Current dispatch: cond_sig_wgan_mc500_d2 (F1 paper-faithful MC=500 +
  # sig_depth=2) — paired with the OPTION 1 patch in
  # src/experiments/adapters/deep_learning/cond_sig_wgan_adapter.py:generate()
  # that retries sampling with a fresh seed when the vendor's generator
  # emits NaN/Inf at later autoregressive steps (caught by
  # validate_artifact in src/utils/artifact_utils.py:51). The previous Wave-3
  # submission (job 694217_2) trained fully but its ARTIFACT was rejected for
  # non-finite values — the trained weights DID save under checkpoints/.
  # Original 3-entry one-per-GAN list was:
  # Restore that original list once all 3 GAN variants have been retrained.
  # Submit with: sbatch --array=0-0%1 --time=4:00:00 --export=ALL,STONKBENCH_FIX_VARIANTS=4 scripts/slurm/final_training.sh
  DL_MODELS=(
    cond_sig_wgan_mc500_d2  # F1 mc=500 d=2; OPTION 1 retry+clamp fix in adapter
  )
elif [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "6" ]]; then
  # Wave 7 GAN architecture-first recovery batch (2026-07-30):
  # 6 variants — 2 per GAN — paired with explicit architecture patches in
  # the adapter layer that close the per-GAN collapse mode surfaced by
  # today's per-channel baseline diagnostic:
  #   QuantGAN (Wave-4 per-asset): mean_std_ratio 1.478 OVERSHOOT.
  #     Fix 1: soft tanh-clamp on per-asset Generator output.
  #     Fix 2: WGAN-GP loss + tanh-clamp (composite).
  #   PCF-GAN (Wave-3 lowdim K=5): mean_std_ratio 0.452 COMPRESSED.
  #     Fix 1: standardize PCA latents before HS critic (fixes PC1 focus).
  #     Fix 2: K=10 + standardize (less compression).
  #   Cond-Sig-WGAN (Wave-3 mc500_d2): mean_std_ratio 0.030 SEVERE.
  #     Fix 1: decouple train q from gen q (train at q=5; vendor AR-FNN
  #       can roll out autoregressively to any generation_length).
  #     Fix 2: qtrain=5 + wider hidden (100,100,100).
  # Submit with: sbatch --array=0-5%6 --time=4:00:00 \
  #   --export=ALL,STONKBENCH_FIX_VARIANTS=6 \
  #   scripts/slurm/final_training.sh
  # Each elapses ~30 min for quantgan (per-asset ~25 sequential trainers),
  # cond_sig_wgan on H100. 4h partition budget is generous headroom.
  DL_MODELS=(
    quantgan_tanhbound             # Wave-7#1: soft tanh clamp per-asset G output
    quantgan_wgangp_tanh           # Wave-7#2: WGAN-GP + tanh-clamp (composite)
    cond_sig_wgan_qtrain5          # Wave-7#5: train q decoupled to 5
    cond_sig_wgan_qtrain5_h100     # Wave-7#6: q=5 + hidden=(100,100,100)
  )
elif [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "5" ]]; then
  # Wave 6 tail-shape recovery (2026-07-30):
  # timegrad_t_noise_df10 — supersedes Wave-5 timegrad_t_noise (df=5) which
  # OVERSHOT the reference distribution. Diagnosis from the Wave-5
  # generated artifact (job 694886, Student-t df=5):
  #   - mean std_ratio    = 1.377   (OVER; baseline 0.681; healthy 0.85-1.15)
  #   - mean |kurt diff|  = 449.9   (baseline 5.3; ch15 alone +624 vs ref)
  #   - mean KS_p         = 0.0027  (tiny lift from baseline 0.0000)
  # Root cause: vendor's epsilon_theta output is an unbounded Conv1d, so the
  # 100-step diffusion reverse kernel compounds extreme Student-t df=5
  # (>3σ events frequent) into chaotic drifty outliers. df=10 is lighter
  # than df=5 but heavier than vendor Gaussian — should pull variance +
  # kurt back toward the reference. Wave-5 candidate (df=5) is preserved
  # in the registry for ablation.
  # The cached-sampler ON-DEVICE variant of OPTION A is in place (df is
  # initialised as a 0-d torch.tensor on the target device; per-call
  # CPU→GPU sync is eliminated) so this run will finish in ~10-15 min on
  # the H100 (the prior job 694745 stalled for 48+ min before the fix).
  # Submit with:
  #   sbatch --array=0-0%1 --time=4:00:00 \
  #     --export=ALL,STONKBENCH_FIX_VARIANTS=5 \
  #     scripts/slurm/final_training.sh
  DL_MODELS=(
    timegrad_t_noise_df10  # Wave 6: Student-t df=10; corrects df=5 OVERSHOOT
  )
elif [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "3" ]]; then
  # Cleanup retrain (2026-07-30) — 5 models, no HP tuning, full vendor_best HP
  # plus 1 minimal-fix variant for the atrophied model.
  # vrnn_klsched_v2 (kl_dim_scale=0.25) SUPERSEDED 2026-07-30 by vrnn_klsched
  # (kl_dim_scale=0.5): v2 produced mean std_ratio=0.640 vs v1's 0.783, so
  # the more-aggressive latent pressure worsened under-dispersion. Removed
  # from this dispatch list — see registry.SUPERSEDED_VARIANTS.
  # Submit with: sbatch --array=0-4%4 --export=ALL,STONKBENCH_FIX_VARIANTS=3 scripts/slurm/final_training.sh
  DL_MODELS=(
    timegrad
    unconditional_tsdiffusion
    conditional_tsdiffusion
    vrnn_klsched               # kl_warmup=20 fights posterior collapse (v1: dim_scale=0.5)
    kalman_vae_safe            # 50-epoch cap fights late-stage loc divergence
  )
elif [[ "${STONKBENCH_FIX_VARIANTS:-0}" == "1" ]]; then
  DL_MODELS=(
    quantgan_clipfix
    quantgan_nstep
    vrnn_epochs50
    timegrad_cells80
    timegrad_lr5e4
    utsd_gclip
    cond_tsd_gclip
  )
else
  DL_MODELS=(
    quantgan
    timegrad
    kalman_vae
    unconditional_tsdiffusion
    conditional_tsdiffusion
    vrnn
    cond_sig_wgan
  )
fi
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#DL_MODELS[@]}" ]]; then
  echo "[FATAL] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= ${#DL_MODELS[@]} (DL_MODELS)." >&2
  exit 2
fi
MODEL="${DL_MODELS[${SLURM_ARRAY_TASK_ID}]}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

# Defaults overridable by env export before `sbatch`.
GENERATION_LENGTH="${GENERATION_LENGTH:-252}"
SEQ_LENGTHS="${SEQ_LENGTHS:-21 42 126}"
# Validate each token is a non-negative integer so a misconfigured SEQ_LENGTHS
# (e.g. "21,42,126" or "21abc") fails loudly here instead of silently in
# argparse. Sequential --seq_lengths flags do NOT accumulate under argparse's
# default `action='store'` + `nargs='+'` — they overwrite. So we validate,
# then emit a single --seq_lengths with space-separated values.
for token in ${SEQ_LENGTHS}; do
  if ! [[ "${token}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] SEQ_LENGTHS entry '${token}' is not a non-negative integer" >&2
    exit 2
  fi
done
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
DEVICE="${STONKBENCH_DEVICE:-cuda}"

echo "=== DL TRAIN ${MODEL} (gen=${GENERATION_LENGTH}, seqs=${SEQ_LENGTHS}, smoke=${STONKBENCH_SMOKE:-0}, dl_set=${STONKBENCH_DL_SET_PATH}) run=${STONKBENCH_RUN_ID} $(date -Is) ==="

# Single --seq_lengths flag with space-separated values matches the parser's
# `nargs='+'` + default `action='store'` contract (programmer_doc:
# `python -m src.experiments.run_final_training --help`).
python -m src.experiments.run_final_training \
  --output_root "${OUTPUT_ROOT}" \
  --generation_length "${GENERATION_LENGTH}" \
  --seq_lengths ${SEQ_LENGTHS} \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --models "${MODEL}" \
  --trim_from_max \
  "${SMOKE_ARGS[@]}"
