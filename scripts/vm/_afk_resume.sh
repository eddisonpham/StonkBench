#!/usr/bin/env bash
# Auto-resume wrapper for the AFK pipeline. Designed to be installed as a
# @reboot cron job so that after the user reboots the VM (to fix a hung NVIDIA
# kernel-mode driver), the AFK pipeline automatically resumes in fresh tmux.
# Because the AFK script has resume-by-skip logic, the 5 stat models already
# trained on CPU are skipped and the 8 DL models re-train on the now-working
# GPU. Subsequent reboots are safe: run_id is timestamped so it never
# collides, and tmux session creation is idempotent.
#
# Install:
#   (crontab -l 2>/dev/null; echo "@reboot /home/phamnhut/StonkBench/scripts/vm/_afk_resume.sh >> /tmp/afk_resume.log 2>&1") | crontab -
# Inspect:
#   crontab -l
# Uninstall:
#   crontab -l | grep -v _afk_resume.sh | crontab -
#
# Logs:
#   /tmp/afk_resume.log                                 — wrapper startup log
#   /tmp/full_pipeline_afk.log                          — main pipeline log
#   /tmp/full_pipeline_afk.<model>.log                  — per-model log
#   /tmp/full_pipeline_afk.done                         — final summary

set -euo pipefail

LOG=/tmp/afk_resume.log
# Cascade: STONKBENCH_RUN_ID > most recent existing > fresh timestamp.
# Reusing the existing run_id is critical so the AFK pipeline's resume-by-skip
# logic actually finds the 4 stat artifacts already on disk under
# outputs/results/<run_id>/<model>/artifacts/. Without this, every reboot
# would create a fresh run_id and we would re-train all 13 models from scratch.
if [ -n "${STONKBENCH_RUN_ID:-}" ]; then
    RUN_ID="${STONKBENCH_RUN_ID}"
    log "Using explicit STONKBENCH_RUN_ID: ${RUN_ID}"
else
    EXISTING_RUN_ID=""
    [ -d outputs/results ] && EXISTING_RUN_ID=$(ls -td outputs/results/*/ 2>/dev/null | head -1 | xargs -n 1 basename)
    if [ -n "${EXISTING_RUN_ID}" ]; then
        RUN_ID="${EXISTING_RUN_ID}"
        log "Reusing existing run_id: ${RUN_ID} (resume-by-skip preserves artifacts)"
    else
        RUN_ID="$(date -u +%Y-%m-%d)_post_reboot_$(date -u +%H%M%S)"
        log "No existing run_id; using fresh: ${RUN_ID}"
    fi
fi

log_ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
# NOTE: single log source. The crontab entry redirects stdout to
# /tmp/afk_resume.log (>> append), so this script writes to stdout only.
# Using `tee -a "$LOG"` here would double-log every line because the crontab
# redirect also writes to the same file.
log()    { printf '[%s] %s\n' "$(log_ts)" "$*"; }

log "============================================================"
log "StonkBench AFK auto-resume triggered"
log "user:       $(whoami)"
log "run_id:     ${RUN_ID}"
log "cwd:        $(pwd)"
log "============================================================"

# 1. Activate conda using the EXACT same pattern as the proven
# run_full_pipeline_afk.sh (which works under cron). Cron has a minimal
# PATH, so we explicitly source the conda activate script.
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/bin/activate" stonk
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/bin/activate" stonk
else
    log "ERROR: conda activate script not found at \$HOME/miniconda3/bin/activate or \$HOME/anaconda3/bin/activate"
    log "ACTION: source \$HOME/miniconda3/bin/activate stonk manually"
    exit 1
fi

# 2. Set up env
export PYTHONPATH="/home/phamnhut/StonkBench:${PYTHONPATH:-}"
export STONKBENCH_RUN_ID="${RUN_ID}"
cd /home/phamnhut/StonkBench || { log "ERROR: cd to project root failed"; exit 1; }

mkdir -p "outputs/results/${RUN_ID}" "outputs/sanity/${RUN_ID}" "outputs/checkpoints/${RUN_ID}"

# 3. Pre-flight GPU sanity (informational only — don't abort).
# Capture stdout and stderr cleanly so the log isn't polluted with
# Traceback noise, and the exit code is from python (not from grep).
set +e
GPU_PROBE=$(python <<'PYEOF' 2>&1
import torch
try:
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    _ = (x @ y).sum().item()
    torch.cuda.synchronize()
    print("GPU ALIVE at boot")
except Exception as e:
    print(f"GPU BROKEN at boot: {type(e).__name__}")
PYEOF
)
GPU_EXIT=$?
set -e
printf '%s\n' "$GPU_PROBE" >> "$LOG"
if [ "$GPU_EXIT" -eq 0 ] && printf '%s' "$GPU_PROBE" | grep -q "GPU ALIVE"; then
    log "GPU is alive — will train all 13 models"
else
    log "GPU is still broken — DL models will fail. User must diagnose further."
fi

# 4. Idempotently launch AFK pipeline in a fresh tmux session.
# If the tmux session already exists (e.g., user rebooted twice in a row),
# kill it first so the new run has a clean session.
if tmux has-session -t post_reboot_pipeline 2>/dev/null; then
    log "Killing leftover post_reboot_pipeline tmux session"
    tmux kill-session -t post_reboot_pipeline
fi
log "Launching AFK pipeline in tmux session 'post_reboot_pipeline'"
HOME=/home/phamnhut tmux new-session -d -s post_reboot_pipeline -x 220 -y 50 \
    "bash /home/phamnhut/StonkBench/scripts/vm/run_full_pipeline_afk.sh 2>&1 | tee /tmp/full_pipeline_afk.log; \
     echo EXIT_CODE=\$? >> /tmp/full_pipeline_afk.log; \
     echo DONE-\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> /tmp/full_pipeline_afk.log"

log "============================================================"
log "AFK pipeline launched; user can monitor with:"
log "  tmux attach -t post_reboot_pipeline"
log "  tail -f /tmp/full_pipeline_afk.log"
log "============================================================"
