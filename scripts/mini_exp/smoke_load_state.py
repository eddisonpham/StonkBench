"""Smoke-test ``adapter.load_state()`` + ``generate()`` for the 7 DL adapters.

Each ``*_seq{L}_final.pt`` checkpoint saved by ``fit`` is rehydrated through the
evaluator's regeneration contract (``can_regenerate_from_checkpoint``) and used
to draw fresh samples. For every cell we report:

  * rebuilt architecture hyper-parameters returned by ``load_state``
  * output shape / finiteness / unique-row count (no silent tiling)
  * denormalized scale vs the shipped artifact of the same ablation
    (``std ratio`` ≈ 1.0 means the rebuilt model reproduces the shipped
    generator; CPU vs GPU RNG makes bit-exact equality impossible, so the
    comparison is distributional)

Examples
--------
# one cell, fast (4 samples)
PYTHONPATH=. STONKBENCH_DEVICE=cpu ~/.venvs/stonkbench/bin/python \
    scripts/mini_exp/smoke_load_state.py --model vrnn_klsched --seq 42 --samples 4

# every model at every seq length, compared against the shipped v2 artifacts
PYTHONPATH=. STONKBENCH_DEVICE=cpu ~/.venvs/stonkbench/bin/python \
    scripts/mini_exp/smoke_load_state.py --seq all --samples 500 --compare
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

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
SEQS = [21, 42, 126, 252]
# Ablations whose checkpoints only exist at seq 252 (trained on W252 only).
SEQ252_ONLY = {"fair_20261004", "latest"}


def _set_dl_set(seq_len: int, ablation: str) -> Path:
    """Point the process at the dl_set that matches the ablation's training."""
    dl_set = REPO_ROOT / "data" / "preprocessed" / f"dl_set_W{seq_len}.pt"
    if not dl_set.exists():
        raise FileNotFoundError(f"dl_set missing: {dl_set}")
    os.environ["STONKBENCH_DL_SET_PATH"] = str(dl_set)
    return dl_set


def _resolve_checkpoint(model: str, seq_len: int, ablation: str) -> tuple[Path, int]:
    """Return (checkpoint path, native generation length)."""
    ckpt_dir = REPO_ROOT / "checkpoints" / ablation / model
    native = 252 if ablation in SEQ252_ONLY else seq_len
    exact = ckpt_dir / f"{model}_seq{seq_len}_final.pt"
    if exact.exists():
        return exact, seq_len
    fallback = ckpt_dir / f"{model}_seq{native}_final.pt"
    if fallback.exists():
        return fallback, native
    raise FileNotFoundError(f"No checkpoint for {model} seq{seq_len} under {ckpt_dir}")


def _load_reference(ablation: str, model: str, seq_len: int) -> torch.Tensor | None:
    """Denormalized shipped artifact for this cell, if it still exists."""
    roots = []
    env_root = os.environ.get("OUTPUT_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(f"/scratch/epham/stonkbench/output/results"))
    for root in roots:
        cand = root / ablation / model / "artifacts" / f"{model}_seq{seq_len}.pt"
        if cand.exists():
            payload = torch.load(cand, map_location="cpu", weights_only=False)
            return payload["data"].float()
    return None


def run_cell(
    model: str,
    seq_len: int,
    ablation: str,
    samples: int,
    seed: int,
    compare: bool,
) -> dict:
    from src.experiments.core.registry import get_adapter
    from src.utils.preprocessed_data_utils import (
        channel_norm_stats,
        denormalize_channels,
        load_dl_set,
    )

    ckpt, native_len = _resolve_checkpoint(model, seq_len, ablation)
    dl_set_path = _set_dl_set(native_len, ablation)

    adapter = get_adapter(model)
    t0 = time.perf_counter()
    info = adapter.load_state([ckpt])
    t_load = time.perf_counter() - t0

    # Generate at the checkpoint's native length, then trim exactly like the
    # pipeline's stitch_sequences (prefix slice for target < base).
    t0 = time.perf_counter()
    out = adapter.generate(num_samples=samples, generation_length=native_len, seed=seed)
    t_gen = time.perf_counter() - t0
    data = out.data
    if data.shape[1] != seq_len:
        data = data[:, :seq_len, :]

    if not torch.isfinite(data).all():
        raise RuntimeError("generated tensor contains NaN/Inf")
    unique = int(torch.unique(data.reshape(data.shape[0], -1), dim=0).shape[0])

    dl_set = load_dl_set(str(dl_set_path))
    mean, std = channel_norm_stats(dl_set)
    denorm = denormalize_channels(data, mean, std).detach().cpu()

    row = {
        "model": model,
        "seq": seq_len,
        "native": native_len,
        "shape": tuple(denorm.shape),
        "unique": unique,
        "finite": True,
        "max_abs": float(denorm.abs().max()),
        "std": float(denorm.std()),
        "load_s": t_load,
        "gen_s": t_gen,
        "ckpt": ckpt.name,
        "info": {k: v for k, v in info.items() if k not in ("checkpoint",)},
    }
    if compare:
        ref = _load_reference(ablation, model, seq_len)
        if ref is None:
            row["ref"] = "missing"
        elif ref.shape[1:] != denorm.shape[1:]:
            row["ref"] = f"shape {tuple(ref.shape)}"
        else:
            # Compare on the overlapping row count so a small smoke run can be
            # checked against a 500-row shipped artifact.
            n = min(int(ref.shape[0]), int(denorm.shape[0]))
            ours, theirs = denorm[:n], ref[:n]
            row["ref_n"] = n
            row["ref_std_ratio"] = float(ours.std() / theirs.std().clamp_min(1e-12))
            row["ref_mean_shift"] = float((ours.mean() - theirs.mean()).abs())
            per_ch = ours.std(dim=(0, 1)) / theirs.std(dim=(0, 1)).clamp_min(1e-12)
            row["ref_ch_ratio_min"] = float(per_ch.min())
            row["ref_ch_ratio_max"] = float(per_ch.max())
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="all",
        help="model key, comma-separated list, or 'all'",
    )
    ap.add_argument("--seq", default="42", help="seq length or 'all'")
    ap.add_argument("--ablation", default="simulation_v2")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", action="store_true", help="compare against shipped artifact")
    args = ap.parse_args()

    models = MODELS if args.model == "all" else [
        m.strip() for m in args.model.split(",") if m.strip()
    ]
    seqs = SEQS if args.seq == "all" else [int(s) for s in args.seq.split(",")]

    failures = 0
    print(
        f"{'model':<30} {'seq':>4} {'nat':>4} {'shape':>16} {'uniq':>5} "
        f"{'std':>10} {'load_s':>7} {'gen_s':>7}  status",
        flush=True,
    )
    for model in models:
        for seq in seqs:
            try:
                row = run_cell(
                    model, seq, args.ablation, args.samples, args.seed, args.compare
                )
            except Exception as exc:  # noqa: BLE001 — smoke harness must keep going
                failures += 1
                print(f"{model:<30} {seq:>4}  FAIL: {exc}", flush=True)
                traceback.print_exc()
                continue
            status = "ok" if row["unique"] == row["shape"][0] else "DUPLICATES"
            if status != "ok":
                failures += 1
            extra = ""
            if "ref_std_ratio" in row:
                extra = (
                    f"  ref_std_ratio={row['ref_std_ratio']:.4f} (n={row['ref_n']})"
                    f" ch=[{row.get('ref_ch_ratio_min', float('nan')):.3f},"
                    f"{row.get('ref_ch_ratio_max', float('nan')):.3f}]"
                    f" mean_shift={row['ref_mean_shift']:.2e}"
                )
            elif "ref" in row:
                extra = f"  ref={row['ref']}"
            print(
                f"{model:<30} {row['seq']:>4} {row['native']:>4} "
                f"{str(row['shape']):>16} {row['unique']:>5} "
                f"{row['std']:>10.5f} {row['load_s']:>7.2f} {row['gen_s']:>7.2f}  "
                f"{status}{extra}",
                flush=True,
            )
    print(f"\n{'ALL CELLS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
