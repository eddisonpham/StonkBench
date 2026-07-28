# TimeGrad STONKBENCH Vendor Patches

kongqi404/timegrad was cloned 2026-07-28 to satisfy the user's "make its adapter"
requirement. The vendor was not designed for univariate per-channel `target_dim=1`
data (its Conv1d kernels assume `length >= 3` along the target_dim axis). This
document records the patches that LANDED on disk in /home/phamnhut/StonkBench with
rationale + reapply strategy + status of the deferred architectural consolidation
(Patch J2).

## Patch A — adapter `_make_gluonts_args` (LANDED)

File: `src/experiments/adapters/deep_learning/timegrad_adapter.py`

Adds `device` parameter to `_make_gluonts_args`. All aux tensors (past_is_pad,
past_time_feat, future_time_feat, target_dimension_indicator) constructed with
explicit `device=dev`. Resolves the cuda/cpu mismatch at
`time_grad_network.py:268` (`torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))`).

## Patch D — adapter `_CONDITIONING_LENGTH` (LANDED, then SUPERSEDED by J1)

Was class-body `_CONDITIONING_LENGTH = 100` (hardcoded), then changed to
`_CONDITIONING_LENGTH = 40` (=`_NUM_CELLS`) at class-body level. Now superseded by
Patch J1 (init-time instance-bound), which is the canonical form.

## Patch G — adapter `_CONDITIONING_LENGTH` static coupling (LANDED, then SUPERSEDED by J1)

Was class-body `_CONDITIONING_LENGTH = _NUM_CELLS` (static). Now superseded by
`self._CONDITIONING_LENGTH = self._NUM_CELLS` at init-time with an assert.

## Patch J1 — adapter init-time coupling (LANDED)

File: `src/experiments/adapters/deep_learning/timegrad_adapter.py`

In `__init__()`, the first 2 lines after `super().__init__()` now read:

```python
self._CONDITIONING_LENGTH = self._NUM_CELLS
assert self._CONDITIONING_LENGTH == self._NUM_CELLS, (
    f"_CONDITIONING_LENGTH ({self._CONDITIONING_LENGTH}) != _NUM_CELLS ({self._NUM_CELLS}). "
    "Re-bind in __init__ before constructing TimeGradTrainingNetwork."
)
```

Replaces static class-body coupling so runtime mutation of `_NUM_CELLS`
propagates. The assert catches any future PR that forgets to keep the invariant.

## Patch B — vendor `input_projection` (LANDED)

File: `src/models/deep_learning/timegrad/epsilon_theta.py`

`nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular")` →
`nn.Conv1d(1, residual_channels, 3, padding=1, padding_mode="zeros")`.

Resolves padding > input_length crash on circular mode at length=1 inputs.

## Patch C — vendor `CondUpsampler.mid` (LANDED, RANK-1 COLLAPSE NOTE)

`CondUpsampler.mid = max(1, target_dim // 2)` (Patch C, left in place). With
`target_dim=1`, `mid=1` produces a rank-1 scalar collapse of the conditioning
signal. The smoke runs (per Patches A + D + G + J1 + B + E + F + H + I) WITHOUT
rank-1 collapse crashing, but the diffusion loses its conditioning signal for
full training. **`Patch J2 is designed to replace this` — see "Deferred Work"
section below.**

## Patch E — vendor `conditioner_projection` (LANDED)

`nn.Conv1d(1, 2 * residual_channels, 1, padding=2, padding_mode="circular")` →
`nn.Conv1d(1, 2 * residual_channels, 1)` (no padding).

Resolves elementwise-add length mismatch with `dilated_conv` output at length=1.

## Patch F — vendor `ResidualBlock.dilated_conv` (LANDED)

`padding_mode="circular"` → `padding_mode="zeros"`. Resolves the "Padding value
causes wrapping around more than once" crash on `dilation=2, padding=2,
input_length=1`.

## Patch H — vendor `skip_projection` (LANDED)

`nn.Conv1d(residual_channels, residual_channels, 3)` →
`nn.Conv1d(residual_channels, residual_channels, 1)` (no padding).

Resolves "Kernel size can't be greater than actual input size" on length=1 input.

## Patch I — vendor `output_projection` (LANDED)

`nn.Conv1d(residual_channels, 1, 3)` → `nn.Conv1d(residual_channels, 1, 1)`.

Same rationale as Patch H.

## Patch J2 — consolidated gated refactor (DEFERRED)

Designed but NOT landed. The design replaces Patches B/C/E/F/H/I with a single
gated refactor:

- `CondUpsampler.mid = max(target_dim, target_dim * 4)` (replaces rank-1 collapse)
- `CondUpsampler` reshapes output to `(B, mid, target_dim)` via reshape
  (NOT view, which fails on non-contiguous layouts)
- `ResidualBlock.dilated_conv` gated: `kernel=3 if target_dim>1 else 1`,
  `padding=dilation if target_dim>1 else 0`, `dilation=dilation if target_dim>1 else 1`
- `ResidualBlock.conditioner_projection = Conv1d(mid, 16, 1)` (consumes
  the rich-mid CondUpsampler output)
- `EpsilonTheta.__init__` threads `target_dim` and `mid` to children; gates
  `input_projection`, `skip_projection`, `output_projection` via
  `kernel_size=self.ks, padding=self.pad`.

**Why deferred:** The regex-DOTALL anchored Python heredoc approach in the prior
session encountered whitespace/comment drift across iterations, and an unbalanced
group-paren regex on the EpsilonTheta.__init__ pattern aborted with set -e before
any write to disk occurred. The Patches A-I give a runnable smoke; Patch J2 is
purely OPTIONAL architectural cleanup (better quality on multi-channel pivot,
richer conditioning).

**How to re-apply cleanly later:**

1. Use `write_file` (single source-of-truth) with the entire new
   `epsilon_theta.py` content.
2. Or use a tightly-bounded sed line-range replacement at file level (NOT
   Python heredoc regex with DOTALL).
3. Or use Python `re.sub` with `(?s)class CondUpsampler\(nn\.Module\):.*?(?=class\s)` — drop unbalanced `\(...\)` escapes.

## Architectural invariants (verifiable via grep)

For `target_dim=1` smoke runs the chain holds:

```
input (B, 1, 252) -> [Conv1d(1, 8, 3, pad=1, zeros)] -> (B, 8, 252)
-> RNN -> (B, subseq_len, 40) -> proj_dist_args Linear(40, 40) -> (B, 40)
-> GaussianDiffusion.log_prob -> reshape (B*T, 1, 40) -> EpsilonTheta
-> input_projection (B*T, 8, 1) -> 8x ResidualBlock stack
-> skip_projection (B*T, 8, 1) -> output_projection (B*T, 1, 1) -> reshape (B, T, 1)
```

The conditioning chain (Patch J2 only):

```
cond_up (B*T, mid=4, target_dim=1) -> conditioner_projection (B*T, 16, 1)
-> elementwise add with dilated_conv output (B*T, 16, 1)
```

## Re-rsync strategy

If `kongqi404/timegrad` is re-cloned and the vendor tree is overwritten, look
for the unique comment stamps:

```
grep -rn STONKBENCH_PATCH_2026-07-28_ src/models/deep_learning/timegrad/epsilon_theta.py
```

If missing on key sites (Patches B/E/F/H/I), re-apply via a single sed block
translating the original vendor lines:

```bash
# Inside epsilon_theta.py
sed -i 's|nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular")|nn.Conv1d(1, residual_channels, 3, padding=1, padding_mode="zeros")|g'
sed -i 's|nn.Conv1d(1, 2 \* residual_channels, 1, padding=2, padding_mode="circular")|nn.Conv1d(1, 2 * residual_channels, 1)|g'
sed -i 's|padding_mode="circular"|padding_mode="zeros"|g'
sed -i 's|self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)|self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)|g'
sed -i 's|self.output_projection = nn.Conv1d(residual_channels, 1, 3)|self.output_projection = nn.Conv1d(residual_channels, 1, 1)|g'
```

## Coverage gap

`adapter.generate(...)` is not exercised by `run_final_training --smoke` itself.
Use `scripts/dev/smoke_timegrad_generate.py` after a smoke completes to verify the
AR-rollout at `unroll_length=1` produces finite, non-zero per-channel samples.
