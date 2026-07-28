#!/usr/bin/env bash
# Standalone per-channel inspection script for the retrained PCF-GAN artifact.
# Run anytime after the artifact lands — does NOT depend on the 30-min poller.
#
# Usage:
#   bash scripts/dev/inspect_pcf_gan.sh
#
# Outputs:
#   - Per-channel table (sim vs gt: std_ratio, s_skew, g_skew, never_pos, in_band)
#   - Validation gate verdict (PASS / FAIL)
#   - Spot-check flags for AMZN, AVGO, CVX, META, NVDA, NFLX

set +e
REPO=/home/phamnhut/StonkBench
ART=$REPO/outputs/results/baseline_2026-07-21_vm/pcf_gan/artifacts/pcf_gan_seq252.pt

echo "=== pcf_gan artifact inspection $(date -u +%FT%TZ) ==="
echo "artifact: $ART"
[ -s "$ART" ] || { echo "  artifact MISSING or 0 bytes — retrain hasn't landed yet"; exit 1; }
echo

cd "$REPO"
source ~/miniconda3/bin/activate stonk 2>/dev/null || true
export PYTHONPATH="$REPO:$PYTHONPATH"

python << PYEOF
import sys, torch, numpy as np
from pathlib import Path
from src.utils.preprocessed_data_utils import (
    load_dl_set, resolve_dl_set_path, denormalize_channels,
)

ART = r'$ART'
print(f'Loading artifact: {ART}')
obj = torch.load(ART, map_location='cpu', weights_only=True)
data = obj['data'].float() if isinstance(obj, dict) else obj.float()
print(f'  sim shape: {tuple(data.shape)}')

dl = load_dl_set(resolve_dl_set_path())
mean, std = dl['channel_mean'], dl['channel_std']
cols = list(dl['feature_columns'])
gt_w = np.asarray(dl['test_windows'], dtype=np.float32)
sim = data.numpy()  # obj['data'] is already in raw space (pipeline.py:170 denormalizes before save); do NOT denorm again.
gt = denormalize_channels(torch.from_numpy(gt_w), mean.float(), std.float()).numpy()
print(f'  gt(test windows) shape: {gt.shape}')
print()

# Spot-check channels — exactly what user flagged
SPOT = {'AMZN', 'AVGO', 'CVX', 'META', 'NVDA', 'NFLX'}

print(f'{"ch":<8}{"sim_std":>10}{"gt_std":>10}{"std_ratio":>11}{"s_skew":>10}{"g_skew":>10}{"never_pos":>11}{"in_band":>10}')
print('-' * 72)
ratios = []
in_band_count = 0
never_pos_count = 0
spot_issues = []
for c in range(len(cols)):
    s = sim[:,:,c].reshape(-1)
    g = gt[:,:,c].reshape(-1)
    s_std = float(s.std())
    g_std = float(g.std())
    sr = s_std / max(g_std, 1e-12)
    s_skew = float(((s - s.mean())**3).mean() / max(s_std**3, 1e-12)) if s_std > 0 else 0.0
    g_skew = float(((g - g.mean())**3).mean() / max(g_std**3, 1e-12)) if g_std > 0 else 0.0
    pos_frac = float((sim[:,:,c] > 0).mean())
    never = pos_frac < 0.01
    in_band = 0.7 <= sr <= 1.3
    if in_band: in_band_count += 1
    if never: never_pos_count += 1
    ratios.append(sr)
    flag_extra = ''
    if cols[c].upper() in SPOT and (never or not in_band):
        flag_extra = ' <-- SPOT FAIL'
        spot_issues.append((cols[c], sr, never, in_band))
    print(f'{cols[c]:<8}{s_std:>10.4f}{g_std:>10.4f}{sr:>11.3f}{s_skew:>10.3f}{g_skew:>10.3f}{str(never):>11}{str(in_band):>10}{flag_extra}')

print()
median_ratio = float(np.median(ratios))
print(f'SUMMARY')
print(f'  median std_ratio:                    {median_ratio:.3f}  (target 0.7-1.3)')
print(f'  channels in [0.7, 1.3] band:         {in_band_count}/25  (target >= 20)')
print(f'  NEVER-POSITIVE channels:             {never_pos_count}/25  (target == 0)')
print()
verdict = (0.7 <= median_ratio <= 1.3) and (never_pos_count == 0) and (in_band_count >= 20)
print(f'GATE VERDICT: {"PASS" if verdict else "FAIL"}')
print()
if spot_issues:
    print('SPOT-CHECK ISSUES (AMZN, AVGO, CVX, META, NVDA, NFLX):')
    for ch, sr, never, in_band in spot_issues:
        print(f'  {ch}: sr={sr:.3f} never_pos={never} in_band={in_band}')
else:
    print('SPOT-CHECK OK: AMZN, AVGO, CVX all in band + zero NEVER-POSITIVE')
print()
if not verdict:
    print('FAIL suggests std_collapse not fully fixed by clamp_k=6 / std_pen floor 0.50 / g_loss 2.0×.')
    print('Suggested next iteration: bump g_loss std_pen multiplier 2.0× → 4.0× AND push min_epochs 80 → 120 in pcf_gan_adapter.py.')
    sys.exit(1)
PYEOF
