# Adapter Benchmark Refactor

This document describes the new adapter-based multivariate benchmark pipeline.

## New entrypoint

- Primary runner: `src/experiments/run_benchmark.py`
- Unified generation script: `src/generation_scripts/generate_data.py` (calls adapter runner directly)

## Standardized contract

- Model input domain: log returns.
- Generated artifact tensor shape: `(R, L, C)` (samples, sequence length, channels/assets).
- Artifacts are routed to: `src/experiments/<model_name>/artifacts`.
- Checkpoints are routed to: `src/experiments/<model_name>/checkpoints`.
- Run logs are routed to: `src/experiments/<model_name>/logs`.

## Adapter registry keys

- `quantgan`
- `timegan`
- `timegrad`
- `timevae`
- `unconditional_tsdiffusion`
- `vrnn`
- `gbm_adapter`, `block_bootstrap` (statistical adapters)

## Smoke test

Run:

```bash
python src/experiments/smoke_test.py
```

The smoke test verifies for each model:

- artifact saved successfully
- tensor shape `(R, L, C)`
- finite float values
- required metadata keys (`asset_columns`, `num_channels`, `model_checkpoint_manifest`)

