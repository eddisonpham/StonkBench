"""Generate multi-batch simulation sequences from the in-repo checkpoints.

Rebuilds a DL adapter from its canonical
``checkpoints/<ablation>/<model>/<model>_seq{L}_final.pt`` through the
``adapter.load_state()`` regeneration contract, draws fresh samples, then
denormalizes and saves them with the same metadata contract the pipeline uses.

The five **statistical** baselines (block / stationary bootstrap, Merton,
DEJD, GARCH(1,1)) have no checkpoints — they are re-``fit`` on the
``stats_set`` train split inside the job and sampled at the requested length;
their artifacts stay in raw space (no denormalization), matching
``run_model_experiment``.

Output layout (eval-compatible: ``unified_evaluator --generated_dir`` expects
``<root>/<model>/artifacts/*.pt``)::

    <out_root>/<ablation>/batch<k>/<model>/artifacts/<model>_seq{L}.pt
    <out_root>/<ablation>/batch<k>/<model>/gen_seq{L>.json       (sidecar)
    <out_root>/<ablation>/ground_truth/ground_truth_seq{L}.pt    (once per ablation)
    <out_root>/<ablation>/ground_truth/statistical/ground_truth_seq{L}.pt

Generation semantics mirror ``src/experiments/core/pipeline.py``:

* native length = the checkpoint's trained window size (``seq{L_ckpt}``),
  so ``simulation_v2`` draws natively at each ``L`` while ``fair_20261004`` /
  ``latest`` (seq-252-only checkpoints) draw 252 steps and are trimmed with
  ``stitch_sequences`` — a pure prefix slice for ``target < base``.
* adapters run in normalized space; the artifact is denormalized with
  ``channel_norm_stats(dl_set)`` of the dl_set the model was trained on
  (``dl_set_W{L_ckpt}.pt``).
* batch seeds are distinct: ``seed = seed_base + 1000 * batch``.

Examples
--------
# one cell, all batches (this is what the array job runs)
PYTHONPATH=. STONKBENCH_DEVICE=cuda python scripts/mini_exp/generate_sequences.py \\
    --ablation simulation_v2 --model timegrad --seq 252 --batches 3 --samples 500

# everything for one ablation, serially
python scripts/mini_exp/generate_sequences.py --ablation simulation_v2 --all --batches 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = [
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "quantgan",
    "timegrad",
    "vrnn_klsched",
    "kalman_vae_safe",
    "cond_sig_wgan",
]
# Statistical baselines: no checkpoints exist for these (they are parametric /
# bootstrap fits), so generate_cell re-fits them on the stats_set train split
# inside the job. Mirrors registry.STATISTICAL_MODEL_KEYS (asserted in main()).
STAT_MODELS = [
    "block_bootstrap",
    "stationary_block_bootstrap",
    "merton_jump_diffusion",
    "de_jump_diffusion",
    "dcc_garch",
]
SEQS = [21, 42, 126, 252]
ABLATIONS = ["simulation_v2", "fair_20261004", "latest"]
# Ablations trained on W252 only: every checkpoint is seq252 and the short
# sequence artifacts are trimmed from the 252 draw (pipeline semantics).
SEQ252_ONLY = frozenset({"fair_20261004", "latest"})
# batch k -> seed = seed_base + SEED_STRIDE * k  (k is 1-based). Keeps the three
# replicate sets independent and away from the original run's seed 42 and the
# probe seeds 43/44/45.
SEED_STRIDE = 1000


def _out_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("STONKBENCH_OUTPUT_ROOT", "").strip()
    base = Path(env) if env else REPO_ROOT / "outputs"
    return (base / "generation").resolve()


def _resolve_checkpoint(model: str, seq_len: int, ablation: str) -> tuple[Path, int]:
    """Return (checkpoint path, native generation length)."""
    ckpt_dir = REPO_ROOT / "checkpoints" / ablation / model
    exact = ckpt_dir / f"{model}_seq{seq_len}_final.pt"
    if exact.exists():
        return exact, seq_len
    fallback = ckpt_dir / f"{model}_seq252_final.pt"
    if fallback.exists():
        return fallback, 252
    raise FileNotFoundError(
        f"No checkpoint for {model} seq{seq_len} under {ckpt_dir}"
    )


def _pin_dl_set(native_len: int) -> Path:
    dl_set = REPO_ROOT / "data" / "preprocessed" / f"dl_set_W{native_len}.pt"
    if not dl_set.exists():
        raise FileNotFoundError(f"dl_set missing: {dl_set}")
    # Must be set before resolve_dl_set_path() reads it (adapters + denorm stats).
    os.environ["STONKBENCH_DL_SET_PATH"] = str(dl_set)
    return dl_set


def _write_ground_truth(
    out_root: Path, ablation: str, native_len: int, seq_len: int, seed: int
) -> Path | None:
    """Denormalized sliding windows of the test split (pipeline parity)."""
    from src.utils.artifact_utils import default_metadata, save_artifact
    from src.utils.preprocessed_data_utils import (
        channel_norm_stats,
        denormalize_channels,
        load_dl_set,
        sliding_window_2d,
    )

    gt_path = out_root / ablation / "ground_truth" / f"ground_truth_seq{seq_len}.pt"
    if gt_path.exists():
        return gt_path
    dl_set = load_dl_set(str(_pin_dl_set(native_len)))
    mean, std = channel_norm_stats(dl_set)
    test_series = denormalize_channels(dl_set["test_series"].float(), mean, std)
    windows = sliding_window_2d(test_series, seq_len, stride=1)
    if windows.shape[0] == 0:
        return None
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: every (model, seq) array task of one ablation races to
    # write the same ground-truth file, so a shared tmp path could interleave.
    tmp = gt_path.with_name(f"{gt_path.stem}.tmp{os.getpid()}.pt")
    meta = default_metadata(
        model_name="ground_truth",
        model_type="ground_truth",
        sequence_length=seq_len,
        num_samples=int(windows.shape[0]),
        seed=seed,
        preprocessing_cfg={
            "dataset_kind": "deep_learning",
            "dl_set_path": str(_pin_dl_set(native_len)),
            "stats_set_path": os.environ.get(
                "STONKBENCH_STATS_SET_PATH",
                str(REPO_ROOT / "data" / "preprocessed" / "statsmodel_set.pt"),
            ),
        },
        extra={
            "num_channels": int(windows.shape[-1]),
            "asset_columns": list(dl_set["feature_columns"]),
            "price_columns": list(dl_set["price_columns"]),
            "is_multivariate": True,
            "ablation": ablation,
        },
    )
    save_artifact(windows, meta, tmp)
    tmp.replace(gt_path)
    return gt_path


def _write_statistical_gt(
    out_root: Path, ablation: str, seq_len: int, seed: int
) -> Path | None:
    """Raw ``stats_set`` test windows — the real-data reference statistical
    models are scored against (the evaluator's ``statistical_test_windows``).

    Same underlying prices as the denormalized dl_set test split (verified to
    1.5e-8), but over the FULL stats test span instead of the window-size
    trimmed dl split, so it yields ``len(test) - L + 1`` windows.
    """
    from src.utils.artifact_utils import default_metadata, save_artifact
    from src.utils.preprocessed_data_utils import (
        load_stats_set,
        resolve_dl_set_path,
        resolve_stats_set_path,
        sliding_window_2d,
    )

    gt_path = (
        out_root
        / ablation
        / "ground_truth"
        / "statistical"
        / f"ground_truth_seq{seq_len}.pt"
    )
    if gt_path.exists():
        return gt_path
    stats_set_path = str(resolve_stats_set_path())
    stats_set = load_stats_set(stats_set_path)
    windows = sliding_window_2d(stats_set["test_series"].float(), seq_len, stride=1)
    if windows.shape[0] == 0:
        return None
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = gt_path.with_name(f"{gt_path.stem}.tmp{os.getpid()}.pt")
    meta = default_metadata(
        model_name="ground_truth",
        model_type="ground_truth",
        sequence_length=seq_len,
        num_samples=int(windows.shape[0]),
        seed=seed,
        preprocessing_cfg={
            "dataset_kind": "statistical",
            "dl_set_path": str(resolve_dl_set_path()),
            "stats_set_path": stats_set_path,
        },
        extra={
            "num_channels": int(windows.shape[-1]),
            "asset_columns": list(stats_set["feature_columns"]),
            "price_columns": list(stats_set["price_columns"]),
            "is_multivariate": True,
            "ablation": ablation,
            "reference": "stats_set_test_series_raw",
        },
    )
    save_artifact(windows, meta, tmp)
    tmp.replace(gt_path)
    return gt_path


def generate_cell(
    ablation: str,
    model: str,
    seq_len: int,
    batches: int,
    samples: int,
    seed_base: int,
    out_root: Path,
) -> List[Dict[str, Any]]:
    from src.experiments.core.registry import get_adapter
    from src.utils.artifact_utils import (
        default_metadata,
        save_artifact,
        stitch_sequences,
    )
    from src.utils.preprocessed_data_utils import (
        channel_norm_stats,
        denormalize_channels,
        load_dl_set,
        resolve_stats_set_path,
    )

    is_stat = model in STAT_MODELS
    load_info: Dict[str, Any] = {}
    fit_info: Dict[str, Any] = {}
    load_sec = 0.0
    mean = std = None

    if is_stat:
        # ---- statistical family: no checkpoints, re-fit on stats_set ----
        from src.experiments.core.contracts import AdapterFitInput
        from src.utils.preprocessed_data_utils import (
            build_batch_from_stats_set,
            load_stats_set,
            resolve_dl_set_path,
        )

        ckpt = None
        native_len = seq_len  # all five statistical adapters generate natively
        stats_set_path = str(resolve_stats_set_path())
        stats_set = load_stats_set(stats_set_path)
        batch = build_batch_from_stats_set(stats_set, generation_length=seq_len)
        preprocessing = {
            "dataset_kind": "statistical",
            "dl_set_path": str(resolve_dl_set_path()),
            "stats_set_path": stats_set_path,
        }
        asset_columns = list(stats_set["feature_columns"])
        price_columns = list(stats_set["price_columns"])

        adapter = get_adapter(model)
        staging = out_root / ablation / ".fit_staging"
        (staging / "checkpoints").mkdir(parents=True, exist_ok=True)
        (staging / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        fit_info = dict(
            adapter.fit(
                AdapterFitInput(
                    batch=batch,
                    sequence_length=seq_len,
                    num_epochs=1,
                    device=str(os.environ.get("STONKBENCH_DEVICE", "cpu")),
                    seed=int(seed_base),
                    metadata={},
                ),
                checkpoints_dir=staging / "checkpoints",
                logs_dir=staging / "logs",
            )
        )
        load_sec = time.perf_counter() - t0
        print(
            f"[gen] {ablation}/{model}/seq{seq_len}: fit(stats_set) "
            f"{load_sec:.2f}s native={native_len} info={fit_info}",
            flush=True,
        )
    else:
        # ---- deep-learning family: rehydrate from the in-repo checkpoint ----
        ckpt, native_len = _resolve_checkpoint(model, seq_len, ablation)
        if ablation in SEQ252_ONLY and native_len != 252:
            raise RuntimeError(
                f"{ablation} must use seq252 checkpoints, got native_len={native_len}"
            )
        dl_set_path = _pin_dl_set(native_len)
        dl_set = load_dl_set(str(dl_set_path))
        mean, std = channel_norm_stats(dl_set)
        preprocessing = {
            "dataset_kind": "deep_learning",
            "dl_set_path": str(dl_set_path),
            "stats_set_path": str(resolve_stats_set_path()),
        }
        asset_columns = list(dl_set["feature_columns"])
        price_columns = list(dl_set["price_columns"])

        adapter = get_adapter(model)
        t0 = time.perf_counter()
        load_info = adapter.load_state([ckpt])
        load_sec = time.perf_counter() - t0
        print(
            f"[gen] {ablation}/{model}/seq{seq_len}: load_state {load_sec:.2f}s "
            f"native={native_len} info={ {k: v for k, v in load_info.items() if k != 'checkpoint'} }",
            flush=True,
        )

    records: List[Dict[str, Any]] = []
    for batch_idx in range(1, batches + 1):
        seed = int(seed_base) + SEED_STRIDE * batch_idx
        out_path = (
            out_root
            / ablation
            / f"batch{batch_idx}"
            / model
            / "artifacts"
            / f"{model}_seq{seq_len}.pt"
        )
        sidecar = out_path.parents[1] / f"gen_seq{seq_len}.json"

        if out_path.exists() and not os.environ.get("STONKBENCH_GEN_FORCE"):
            print(f"[gen] skip existing {out_path}", flush=True)
            records.append({"path": str(out_path), "skipped": True, "seed": seed})
            continue

        t0 = time.perf_counter()
        out = adapter.generate(
            num_samples=samples, generation_length=native_len, seed=seed
        )
        gen_sec = time.perf_counter() - t0

        data = out.data
        stitched = data.shape[1] != seq_len
        if stitched:
            data = stitch_sequences(data, seq_len, seed=seed)
        if data.shape[0] != samples:
            raise RuntimeError(
                f"{model} seq{seq_len} batch{batch_idx}: expected {samples} "
                f"rows, got {data.shape[0]}"
            )
        if not torch.isfinite(data).all():
            raise RuntimeError(f"{model} seq{seq_len}: non-finite samples")
        n_unique = int(torch.unique(data.reshape(data.shape[0], -1), dim=0).shape[0])

        if is_stat:
            # stats_set is already raw space — the pipeline does not denormalize
            # statistical artifacts (see run_model_experiment).
            saved = data.detach().cpu().float()
        else:
            saved = denormalize_channels(data, mean, std).detach().cpu().float()

        metadata = default_metadata(
            model_name=model,
            model_type="statistical" if is_stat else "deep_learning",
            sequence_length=seq_len,
            num_samples=samples,
            seed=seed,
            preprocessing_cfg=preprocessing,
            extra={
                **{k: v for k, v in load_info.items() if k != "checkpoint"},
                **fit_info,
                **out.extra_metadata,
                "num_channels": int(saved.shape[-1]),
                "asset_columns": asset_columns,
                "price_columns": price_columns,
                "is_multivariate": True,
                # DL: the dl_set window_size the checkpoint trained on.
                # Statistical: the stats_set train span is fitted in full
                # (2221 steps); native_len is what the pipeline records here.
                "train_sequence_length": int(native_len),
                # Some adapters historically returned an empty checkpoint list
                # from generate(); fall back to the checkpoint we just loaded
                # so the artifact is always regenerable from its manifest.
                # Statistical adapters have no checkpoints at all — they are
                # re-fitted from stats_set (flagged below), so an empty manifest
                # is correct there.
                "model_checkpoint_manifest": [
                    str(p) for p in ((out.checkpoints or ([ckpt] if ckpt else [])))
                ],
                "ablation": ablation,
                "model_family": "statistical" if is_stat else "deep_learning",
                "generation_batch": batch_idx,
                "native_length": int(native_len),
                "stitched_from_native": bool(stitched),
                "restored_from_checkpoint": not is_stat,
                "fit_from_stats_set": is_stat,
                "checkpoint_load_seconds": round(load_sec, 3),
            },
        )
        # Atomic publish: a job killed mid-save must not leave a truncated
        # artifact that the skip-existing check would then treat as done.
        tmp_path = out_path.with_name(f"{out_path.stem}.tmp{os.getpid()}.pt")
        save_artifact(saved, metadata, tmp_path)
        tmp_path.replace(out_path)

        record = {
            "ablation": ablation,
            "model": model,
            "model_family": "statistical" if is_stat else "deep_learning",
            "sequence_length": seq_len,
            "native_length": int(native_len),
            "batch": batch_idx,
            "seed": seed,
            "num_samples": samples,
            "unique_samples": n_unique,
            "shape": list(saved.shape),
            "std": float(saved.std()),
            "max_abs": float(saved.abs().max()),
            "checkpoint": str(ckpt) if ckpt else None,
            "fit_seconds": round(load_sec, 2),
            "generate_seconds": round(gen_sec, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "path": str(out_path),
        }
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
        sidecar_tmp.replace(sidecar)
        print(
            f"[gen] {ablation}/{model}/seq{seq_len}/batch{batch_idx}: "
            f"{tuple(saved.shape)} unique={n_unique} std={saved.std():.5f} "
            f"gen={gen_sec:.1f}s -> {out_path}",
            flush=True,
        )
        records.append(record)

    # Real-data reference for this ablation. DL cells pin the dl_set matching
    # the checkpoint's native window; statistical cells score against the full
    # stats_set test span instead.
    if is_stat:
        _write_statistical_gt(out_root, ablation, seq_len, seed_base)
    else:
        _write_ground_truth(out_root, ablation, native_len, seq_len, seed_base)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", default="simulation_v2", choices=ABLATIONS)
    ap.add_argument(
        "--model",
        default=None,
        help="model key, comma-separated list, 'all', or (with --all) every model",
    )
    ap.add_argument("--seq", type=int, default=None, help="seq length (or --all)")
    ap.add_argument("--all", action="store_true", help="run every model × seq cell")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    out_root = _out_root(args.out_root)
    from src.experiments.core.registry import STATISTICAL_MODEL_KEYS

    if set(STAT_MODELS) != set(STATISTICAL_MODEL_KEYS):
        raise SystemExit(
            "STAT_MODELS out of sync with registry.STATISTICAL_MODEL_KEYS: "
            f"{sorted(STAT_MODELS)} vs {sorted(STATISTICAL_MODEL_KEYS)}"
        )
    models = (
        MODELS + STAT_MODELS
        if (args.all or args.model in (None, "all"))
        else [m.strip() for m in args.model.split(",") if m.strip()]
    )
    seqs = SEQS if (args.all or args.seq in (None, 0)) else [args.seq]
    if not args.all and args.model in (None, "all") and args.seq is None:
        raise SystemExit("Pass --model/--seq or --all")

    failures = 0
    for model in models:
        for seq in seqs:
            try:
                generate_cell(
                    ablation=args.ablation,
                    model=model,
                    seq_len=seq,
                    batches=args.batches,
                    samples=args.samples,
                    seed_base=args.seed_base,
                    out_root=out_root,
                )
            except Exception as exc:  # noqa: BLE001 — keep the sweep going
                failures += 1
                import traceback

                print(f"[gen-FAIL] {args.ablation}/{model}/seq{seq}: {exc}", flush=True)
                traceback.print_exc()
    print(
        f"[gen] done ablation={args.ablation} out_root={out_root} "
        f"failures={failures}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
