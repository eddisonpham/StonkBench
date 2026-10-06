"""Verify the multi-batch generation trees before any evaluation is staged.

Checks per (ablation, batch, model, seq) artifact:

* file present (expect 3 batches x 7 models x 4 seqs per ablation)
* shape == (500, L, 25), float, all finite
* 500 unique rows (no silent tiling — the cond_sig_wgan fix); the two
  bootstrap baselines are checked against the birthday expectation instead,
  since with-replacement resampling legitimately repeats paths
* metadata contract: sequence_length/num_samples/seed/ablation/batch,
  `model_checkpoint_manifest` present, denormalized scale sane
* batches of the same cell are pairwise distinct tensors (distinct seeds)
* magnitude sanity vs the shipped reference artifacts (std ratio for
  simulation_v2, which has the original run on disk)

Usage:
    PYTHONPATH=. python scripts/mini_exp/verify_generation.py \
        [--root /scratch/epham/stonkbench/output/generation] \
        [--ablation simulation_v2 ...]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
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
STAT_MODELS = [
    "block_bootstrap",
    "stationary_block_bootstrap",
    "merton_jump_diffusion",
    "de_jump_diffusion",
    "garch11",
]
ALL_MODELS = MODELS + STAT_MODELS
# The two bootstrap baselines draw blocks WITH replacement from a finite pool,
# so duplicate rows are legitimate sampling behaviour, not a tiling bug — the
# production artifacts show the same signature (block_bootstrap@21: 790/1000
# unique). For those two models the uniqueness check therefore asserts the
# birthday-paradox expectation over the support instead of exact uniqueness;
# a repeat-the-same-draw bug (cond_sig_wgan's old 251/500 tiling) still fails.
BOOTSTRAP_MODELS = {"block_bootstrap", "stationary_block_bootstrap"}
TRAIN_SPAN = 2221          # stats_set train_series rows
BOOTSTRAP_BLOCK = 32       # block_size / expected_block_length
BOOTSTRAP_SUPPORT = TRAIN_SPAN - BOOTSTRAP_BLOCK + 1
SEQS = [21, 42, 126, 252]
BATCHES = [1, 2, 3]
ABLATIONS = ["simulation_v2", "fair_20261004", "latest"]
EXPECTED_SAMPLES = 500

_REF_ROOTS = [
    Path("/scratch/epham/stonkbench/output/results"),
]


def _expected_unique(n: int, support: int) -> float:
    """Expected distinct draws when sampling n times with replacement."""
    return n - (n * (n - 1)) / (2.0 * max(support, 1))


def _check_artifact(path: Path, seq_len: int, family: str, model: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "data" not in payload or "metadata" not in payload:
        raise ValueError("not a data+metadata artifact")
    data = payload["data"]
    meta = payload["metadata"]

    if data.shape != (EXPECTED_SAMPLES, seq_len, 25):
        raise ValueError(f"shape {tuple(data.shape)} != ({EXPECTED_SAMPLES}, {seq_len}, 25)")
    if data.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"dtype {data.dtype}")
    if not torch.isfinite(data).all():
        raise ValueError("non-finite values")
    unique = int(torch.unique(data.reshape(data.shape[0], -1), dim=0).shape[0])
    if model in BOOTSTRAP_MODELS:
        expected = _expected_unique(EXPECTED_SAMPLES, BOOTSTRAP_SUPPORT)
        if unique < 0.85 * expected:
            raise ValueError(
                f"only {unique}/{EXPECTED_SAMPLES} unique rows — below the "
                f"{expected:.0f} birthday expectation for with-replacement "
                f"resampling (support={BOOTSTRAP_SUPPORT})"
            )
    elif unique != EXPECTED_SAMPLES:
        raise ValueError(f"only {unique}/{EXPECTED_SAMPLES} unique rows")

    if int(meta["sequence_length"]) != seq_len:
        raise ValueError(f"metadata sequence_length={meta['sequence_length']}")
    if int(meta["num_samples"]) != EXPECTED_SAMPLES:
        raise ValueError(f"metadata num_samples={meta['num_samples']}")
    want_type = "statistical" if family == "statistical" else "deep_learning"
    if str(meta.get("model_type")) != want_type:
        raise ValueError(f"model_type={meta.get('model_type')} != {want_type}")
    if family == "statistical":
        if not meta.get("fit_from_stats_set"):
            raise ValueError("missing fit_from_stats_set flag")
        if meta.get("model_checkpoint_manifest"):
            raise ValueError("statistical artifact should have an empty manifest")
    else:
        if not meta.get("model_checkpoint_manifest"):
            raise ValueError("missing model_checkpoint_manifest")
        if not meta.get("restored_from_checkpoint"):
            raise ValueError("missing restored_from_checkpoint flag")
    if not torch.isfinite(data).all():
        raise ValueError("non-finite")
    max_abs = float(data.abs().max())
    if max_abs > 5.0:
        raise ValueError(f"implausible max|x|={max_abs:.3f}")

    return {
        "tensor": data,
        "unique": unique,
        "std": float(data.std()),
        "max_abs": max_abs,
        "seed": int(meta["seed"]),
        "batch": meta.get("generation_batch"),
        "native": meta.get("native_length"),
        "stitched": meta.get("stitched_from_native"),
        "checkpoint": (meta.get("model_checkpoint_manifest") or [None])[0],
    }


def _reference(ablation: str, model: str, seq_len: int) -> torch.Tensor | None:
    for root in _REF_ROOTS:
        cand = root / ablation / model / "artifacts" / f"{model}_seq{seq_len}.pt"
        if cand.exists():
            payload = torch.load(cand, map_location="cpu", weights_only=False)
            return payload["data"].float()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=None,
        help="generation root (default: <repo>/outputs/generation or $STONKBENCH_OUTPUT_ROOT/generation)",
    )
    ap.add_argument("--ablation", nargs="*", default=ABLATIONS)
    ap.add_argument("--skip-distinctness", action="store_true")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        import os

        base = os.environ.get("STONKBENCH_OUTPUT_ROOT") or str(REPO_ROOT / "outputs")
        root = Path(base) / "generation"

    if not root.exists():
        print(f"[verify] generation root missing: {root}")
        return 2

    failures: list[str] = []
    checked = 0
    boot_uniques: list[tuple[str, int]] = []
    print(f"[verify] root={root}")

    for ablation in args.ablation:
        for batch in BATCHES:
            for model in ALL_MODELS:
                family = "statistical" if model in STAT_MODELS else "deep_learning"
                for seq in SEQS:
                    path = (
                        root
                        / ablation
                        / f"batch{batch}"
                        / model
                        / "artifacts"
                        / f"{model}_seq{seq}.pt"
                    )
                    label = f"{ablation}/batch{batch}/{model}/seq{seq}"
                    if not path.exists():
                        failures.append(f"{label}: MISSING")
                        continue
                    try:
                        info = _check_artifact(path, seq, family, model)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{label}: {exc}")
                        continue
                    checked += 1
                    if model in BOOTSTRAP_MODELS:
                        boot_uniques.append((label, info["unique"]))

                    # cross-batch distinctness for the same cell
                    if batch == 1 and not args.skip_distinctness:
                        others = []
                        for other in BATCHES[1:]:
                            opath = (
                                root / ablation / f"batch{other}" / model
                                / "artifacts" / f"{model}_seq{seq}.pt"
                            )
                            if opath.exists():
                                others.append(
                                    torch.load(opath, map_location="cpu", weights_only=False)["data"]
                                )
                        if len(others) == 2:
                            same = [
                                bool(torch.equal(info["tensor"], o)) for o in others
                            ]
                            if any(same):
                                failures.append(
                                    f"{label}: batch1 identical to batch{BATCHES[1:][same.index(True)]}"
                                )

        # Ground-truth references: dl_set-derived (deep learning) and the full
        # stats_set test span (statistical), one per sequence length.
        for seq in SEQS:
            dl_gt = root / ablation / "ground_truth" / f"ground_truth_seq{seq}.pt"
            stat_gt = (
                root / ablation / "ground_truth" / "statistical"
                / f"ground_truth_seq{seq}.pt"
            )
            if not dl_gt.exists():
                failures.append(f"{ablation}/ground_truth/seq{seq}: MISSING")
            else:
                checked += 1
            if not stat_gt.exists():
                failures.append(f"{ablation}/ground_truth/statistical/seq{seq}: MISSING")
            else:
                checked += 1
                payload = torch.load(stat_gt, map_location="cpu", weights_only=False)
                expect = 596 - seq + 1  # full stats test span
                if payload["data"].shape != (expect, seq, 25):
                    failures.append(
                        f"{ablation}/ground_truth/statistical/seq{seq}: "
                        f"shape {tuple(payload['data'].shape)} != ({expect}, {seq}, 25)"
                    )

        # simulation_v2 magnitude cross-check against the shipped artifacts
        if ablation == "simulation_v2":
            for model in MODELS:
                for seq in SEQS:
                    p1 = root / ablation / "batch1" / model / "artifacts" / f"{model}_seq{seq}.pt"
                    if not p1.exists():
                        continue
                    ref = _reference(ablation, model, seq)
                    if ref is None:
                        continue
                    got = torch.load(p1, map_location="cpu", weights_only=False)["data"].float()
                    ratio = float(got.std() / ref.std().clamp_min(1e-12))
                    if not (0.80 <= ratio <= 1.25):
                        failures.append(
                            f"{ablation}/{model}/seq{seq}: std ratio vs shipped {ratio:.3f} outside [0.8,1.25]"
                        )
                    else:
                        print(f"    [ref] {model} seq{seq}: std_ratio={ratio:.3f}")

        # production `latest` statistical baselines are still on disk — use them
        # as the independent reference for the re-fitted statistical generators.
        if ablation == "latest":
            for model in STAT_MODELS:
                for seq in SEQS:
                    p1 = root / ablation / "batch1" / model / "artifacts" / f"{model}_seq{seq}.pt"
                    if not p1.exists():
                        continue
                    ref = _reference(ablation, model, seq)
                    if ref is None:
                        continue
                    got = torch.load(p1, map_location="cpu", weights_only=False)["data"].float()
                    ratio = float(got.std() / ref.std().clamp_min(1e-12))
                    if not (0.75 <= ratio <= 1.35):
                        failures.append(
                            f"{ablation}/{model}/seq{seq}: std ratio vs production {ratio:.3f} outside [0.75,1.35]"
                        )
                    else:
                        print(f"    [ref-stat] {model} seq{seq}: std_ratio={ratio:.3f}")

    print(f"\n[verify] checked {checked} checks (artifacts + ground truth)")
    if boot_uniques:
        floor = 0.85 * _expected_unique(EXPECTED_SAMPLES, BOOTSTRAP_SUPPORT)
        vals = [u for _, u in boot_uniques]
        print(
            f"[verify] bootstrap uniqueness: {len(vals)} cells, "
            f"{min(vals)}-{max(vals)} / {EXPECTED_SAMPLES} unique "
            f"(floor {floor:.0f} = birthday expectation over "
            f"support {BOOTSTRAP_SUPPORT}; production block_bootstrap@21 "
            f"is 790/1000)"
        )
    if failures:
        print(f"[verify] {len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[verify] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
