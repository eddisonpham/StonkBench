#!/usr/bin/env python
"""Mini DL validation run: short-budget train + generate + fidelity vs GT.

One model per invocation (Slurm array parallelism); all outputs land in the
shared run directory ``<out_root>/<run_id>/`` so every model can be monitored
simultaneously (see exp_logger.refresh_report + ``monitor.sh``).

Behavior notes:
- Training budget is deliberately SHORT (--epochs, early stopping disabled by
  default) so we can watch full training curves on the 2h debug partition.
- Nothing writes to the production outputs/ tree: checkpoints, curves, plots
  and reports stay under the mini run dir.
- Live per-epoch events stream to ``events/<model>.jsonl`` via
  src.utils.epoch_events.emit_epoch (env-gated, behavior-neutral).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp_logger import MiniLogger, compute_fidelity, refresh_report  # noqa: E402

from src.experiments.core.contracts import AdapterFitInput  # noqa: E402
from src.experiments.core.registry import (  # noqa: E402
    VARIANT_TO_BASE,
    get_adapter,
)
from src.experiments.hp_configs import (  # noqa: E402
    FULL_TRAIN_EPOCHS,
    MODEL_HP_CONFIGS,
    full_train_metadata,
)
from src.utils.artifact_utils import stitch_sequences  # noqa: E402
from src.utils.device import device_to_str, get_device  # noqa: E402
from src.utils.preprocessed_data_utils import (  # noqa: E402
    build_batch_from_dl_set,
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)

DEFAULT_DL_MODELS = [
    "quantgan",
    "timegrad",
    "kalman_vae_safe",
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "vrnn_klsched",
    "cond_sig_wgan",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=DEFAULT_DL_MODELS)
    p.add_argument("--run_id", required=True)
    p.add_argument(
        "--out_root",
        default=os.environ.get("STONKBENCH_MINI_ROOT", "/home/epham/StonkBench/outputs/mini_exp"),
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop patience. Default epochs+10 = effectively disabled, "
        "so full training curves are recorded.",
    )
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seq_lengths", default="21,42,126,252")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="cond_sig_wgan step budget (step-based trainer, not epoch-based).",
    )
    p.add_argument("--skip_plots", action="store_true")
    return p.parse_args()


def build_metadata(model_key: str, epochs: int, patience: int, native_len: int, steps: int):
    """Vendor-best training metadata with mini-budget overrides."""
    base_key = VARIANT_TO_BASE.get(model_key, model_key)
    cfgs = MODEL_HP_CONFIGS.get(model_key) or MODEL_HP_CONFIGS[base_key]
    cfg = next((c for c in cfgs if getattr(c, "is_vendor_default", False)), cfgs[0])
    best = dataclasses.asdict(cfg)
    best["mean_best_val_loss"] = 0.0
    metadata = full_train_metadata(model_key, {"best_config": best})
    metadata.update(
        {
            "max_epochs": int(epochs),
            "patience": int(patience),
            "model_key": model_key,
            "generation_length": int(native_len),
        }
    )
    if model_key == "cond_sig_wgan":
        # Step-based trainer: explicit budget; full_train epoch cap only
        # gates the sub-10-epoch smoke clamp inside the adapter.
        metadata["cond_sig_wgan_steps"] = int(steps)
        metadata["max_epochs"] = int(FULL_TRAIN_EPOCHS.get(model_key, 100))
    return metadata


def main() -> int:
    args = parse_args()
    run_dir = Path(args.out_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Live events -> per-model file inside the run dir (same path the Slurm
    # script exports, so both conventions converge).
    os.environ.setdefault(
        "STONKBENCH_MINI_EVENTS", str(run_dir / "events" / f"{args.model}.jsonl")
    )
    (run_dir / "events").mkdir(parents=True, exist_ok=True)

    logger = MiniLogger(run_dir, args.model)
    device = device_to_str(get_device(args.device))
    seq_lengths = sorted(
        {int(x) for x in str(args.seq_lengths).split(",") if str(x).strip()}
    )
    patience = args.patience if args.patience is not None else args.epochs + 10

    print(f"[mini] model={args.model} run={args.run_id} device={device} "
          f"epochs={args.epochs} patience={patience} n={args.num_samples} "
          f"seqs={seq_lengths}", flush=True)
    logger.event("run_start", device=device, epochs=args.epochs, patience=patience,
                 num_samples=args.num_samples, seq_lengths=seq_lengths, seed=args.seed)

    # ---- data -----------------------------------------------------------
    dl_set = load_dl_set(resolve_dl_set_path())
    native_len = int(dl_set["window_size"])
    batch = build_batch_from_dl_set(dl_set, generation_length=native_len)
    norm_stats = channel_norm_stats(dl_set)
    if norm_stats is None:
        raise RuntimeError("dl_set has no channel_mean/channel_std; cannot denormalize.")
    mean, std = norm_stats
    asset_names = list(batch.asset_columns)
    print(f"[mini] dl_set window={native_len} train={tuple(batch.train_windows.shape)} "
          f"valid={tuple(batch.valid_windows.shape)} test={tuple(batch.test.shape)}", flush=True)

    metadata = build_metadata(args.model, args.epochs, patience, native_len, args.steps)
    print(f"[mini] metadata max_epochs={metadata['max_epochs']} "
          f"lr={metadata['learning_rate']} bs={metadata['batch_size']} "
          f"patience={metadata['patience']}", flush=True)

    # ---- fit ------------------------------------------------------------
    adapter = get_adapter(args.model)
    ckpt_dir = run_dir / "checkpoints" / args.model
    log_dir = run_dir / "logs" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fit_input = AdapterFitInput(
        batch=batch,
        sequence_length=native_len,
        num_epochs=args.epochs,
        device=device,
        seed=args.seed,
        metadata=metadata,
    )
    t0 = time.perf_counter()
    try:
        fit_info = adapter.fit(fit_input, checkpoints_dir=ckpt_dir, logs_dir=log_dir)
    except Exception as exc:  # noqa: BLE001 — log the failure, then fail loudly
        logger.event("error", stage="fit", error=str(exc), traceback=traceback.format_exc())
        print(f"[mini][ERROR] {args.model} fit failed: {exc}", flush=True)
        traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - t0
    logger.log_fit(fit_info, elapsed)
    print(f"[mini] fit done in {elapsed:.1f}s best_val={fit_info.get('best_val_loss')} "
          f"best_epoch={fit_info.get('best_epoch')} "
          f"stopped_early={fit_info.get('stopped_early')}", flush=True)

    step_losses = getattr(adapter, "step_losses", None)
    if step_losses:
        logger.log_step_losses(step_losses)
    if not args.skip_plots:
        try:
            logger.plot_curves(fit_info)
        except Exception as exc:  # noqa: BLE001
            print(f"[mini][WARN] curve plot failed: {exc}", flush=True)

    # ---- generate + fidelity --------------------------------------------
    t0 = time.perf_counter()
    try:
        gen_out = adapter.generate(
            num_samples=args.num_samples, generation_length=native_len, seed=args.seed
        )
    except Exception as exc:  # noqa: BLE001
        logger.event("error", stage="generate", error=str(exc), traceback=traceback.format_exc())
        print(f"[mini][ERROR] {args.model} generate failed: {exc}", flush=True)
        traceback.print_exc()
        return 1
    gen_elapsed = time.perf_counter() - t0
    z = gen_out.data.detach().cpu().float()
    print(f"[mini] generate {tuple(z.shape)} in {gen_elapsed:.1f}s "
          f"nonfinite={float((~torch.isfinite(z)).float().mean()):.4f}", flush=True)

    test_raw = denormalize_channels(batch.test.float(), mean, std)
    for seq_len in seq_lengths:
        z_seq = z if z.shape[1] == seq_len else stitch_sequences(z, seq_len, seed=args.seed)
        gen_raw = denormalize_channels(z_seq, mean, std).numpy()
        gt_windows = sliding_window_2d(test_raw, seq_len, stride=1).numpy()
        if gt_windows.shape[0] == 0:
            print(f"[mini][WARN] no GT windows for seq{seq_len}; skipping", flush=True)
            continue
        metrics = compute_fidelity(gen_raw, gt_windows)
        logger.log_fidelity(seq_len, metrics)
        print(
            f"[mini] seq{seq_len}: std_ratio={metrics['std_ratio_mean']:.3f} "
            f"healthy={metrics['std_ratio_healthy_frac']*100:.0f}% "
            f"kurt_diff={metrics['kurt_abs_diff']:.2f} lag1_diff={metrics['lag1_abs_diff']:.3f} "
            f"corr_diff={metrics['corr_rel_frob']:.3f} KS={metrics['ks_mean']:.3f} "
            f"nonfinite={metrics['nonfinite_frac']:.4f} "
            f"(n_gen={metrics['n_gen']}, n_gt={metrics['n_gt']})",
            flush=True,
        )
        if not args.skip_plots:
            try:
                logger.plot_fidelity(seq_len, metrics, gen_raw, gt_windows, asset_names)
            except Exception as exc:  # noqa: BLE001
                print(f"[mini][WARN] fidelity plot failed: {exc}", flush=True)

    logger.event("run_complete", generate_seconds=gen_elapsed)
    try:
        report = refresh_report(run_dir)
        print(f"[mini] report refreshed: {report}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mini][WARN] refresh_report failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
