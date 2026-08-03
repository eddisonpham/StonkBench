#!/usr/bin/env python3
"""Monitor SigWGAN fix variant jobs and run per-channel diagnostics on .pt artifacts as they land.

Usage:
  python scripts/dev/sigwgan_fix_monitor.py \
    --job-ids 702891,702892,702893,702894,702895,702896 \
    --output-root /scratch/epham/stonkbench/gan_tests/sigwgan_fixes \
    --poll-interval 300 \
    --max-wait 3600
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

VARIANTS = [
    # Round 1 (2026-08-01): calibration-side knobs — all plateaued.
    "csigwgan_ols_s1",
    "csigwgan_noscale",
    "csigwgan_ridge001",
    "csigwgan_mc500_lr2",
    "csigwgan_q50",
    "csigwgan_ols_s1_ns",
    # Round 2 (2026-08-03): latent-noise amplification + var-reg loss.
    "csigwgan_noise5",
    "csigwgan_noise10",
    "csigwgan_varreg1",
    "csigwgan_noise5_varreg1",
    "csigwgan_q50_noise5",
    "csigwgan_noise5_lr3",
]


def check_job_status(job_ids: list[str]) -> dict[str, str]:
    """Run sacct and return {task_id: state}."""
    ids = ",".join(job_ids)
    result = subprocess.run(
        ["sacct", "-j", ids, "-X", "-o", "JobID,State", "--noheader"],
        capture_output=True, text=True, timeout=15,
    )
    status = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            job_id = parts[0].strip()
            state = parts[1].strip()
            status[job_id] = state
    return status


def load_ground_truth() -> np.ndarray:
    """Load ground truth from the main results/latest (or fallback from scratch)."""
    gt_paths = [
        Path("/scratch/epham/stonkbench/output/results/latest/ground_truth/ground_truth_seq252.pt"),
        Path("/home/epham/StonkBench/outputs/results/latest/ground_truth/ground_truth_seq252.pt"),
    ]
    for p in gt_paths:
        if p.exists():
            d = torch.load(str(p), map_location="cpu")
            if isinstance(d, dict):
                return d["data"].float().numpy()
            return d.float().numpy()
    raise FileNotFoundError("Ground truth not found")


def run_diagnostics(pt_path: Path, gt: np.ndarray) -> dict:
    """Run per-channel std_ratio + KS diagnostic on a .pt artifact vs ground truth."""
    d = torch.load(str(pt_path), map_location="cpu")
    pt_data = d["data"].float().numpy() if isinstance(d, dict) else d.float().numpy()

    # Per-channel std_ratio
    pt_std = pt_data.reshape(-1, pt_data.shape[-1]).std(axis=0)
    gt_std = gt.reshape(-1, gt.shape[-1]).std(axis=0)
    ratios = pt_std / (gt_std + 1e-12)

    # Per-channel KS
    ks_ps = []
    for c in range(pt_data.shape[-1]):
        _, p = ks_2samp(pt_data[:, :, c].ravel(), gt[:, :, c].ravel())
        ks_ps.append(p)

    return {
        "std_ratio_mean": float(ratios.mean()),
        "std_ratio_min": float(ratios.min()),
        "std_ratio_max": float(ratios.max()),
        "ks_p_mean": float(np.mean(ks_ps)),
        "ks_p_min": float(np.min(ks_ps)),
        "nan_count": int(np.isnan(pt_data).sum()),
        "pt_range": [float(pt_data.min()), float(pt_data.max())],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-ids", required=True, help="Comma-separated Slurm job IDs")
    parser.add_argument("--output-root", default="/scratch/epham/stonkbench/gan_tests/sigwgan_fixes")
    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-wait", type=int, default=3600)
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="Comma-separated variant names to watch")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    job_ids = [jid.strip() for jid in args.job_ids.split(",")]
    output_root = Path(args.output_root)

    print(f"=== SigWGAN Fix Monitor ===")
    print(f"Jobs: {job_ids}")
    print(f"Output root: {output_root}")
    print(f"Poll interval: {args.poll_interval}s, max wait: {args.max_wait}s")
    print()

    try:
        gt = load_ground_truth()
        print(f"Ground truth loaded: shape={gt.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    start_time = time.time()
    results: dict[str, dict] = {}

    while time.time() - start_time < args.max_wait:
        status = check_job_status(job_ids)
        active = [j for j, s in status.items() if s in ("RUNNING", "PENDING", "COMPLETING")]
        completed = [j for j, s in status.items() if s == "COMPLETED"]
        failed = [j for j, s in status.items() if s in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY")]

        print(f"\n[{time.strftime('%H:%M:%S')}] Active: {len(active)}, Done: {len(completed)}, Failed: {len(failed)}")
        if active:
            print(f"  Active: {', '.join(active)}")
        if failed:
            print(f"  FAILED: {', '.join(failed)}")

        # Check for new artifacts
        for variant in variants:
            if variant in results:
                continue  # already diagnosed
            # Try to find seq252 artifact
            search_paths = [
                output_root / "results" / "latest" / variant / "artifacts" / f"{variant}_seq252.pt",
                output_root / variant / "artifacts" / f"{variant}_seq252.pt",
                Path("/scratch/epham/stonkbench/output/results/latest") / variant / "artifacts" / f"{variant}_seq252.pt",
            ]
            for pt_path in search_paths:
                if pt_path.exists():
                    try:
                        diag = run_diagnostics(pt_path, gt)
                        results[variant] = diag
                        verdict = (
                            "HEALTHY" if 0.7 <= diag["std_ratio_mean"] <= 1.5
                            else "LOW_VAR" if diag["std_ratio_mean"] < 0.7
                            else "HIGH_VAR" if diag["std_ratio_mean"] > 1.5
                            else "COLLAPSED" if diag["std_ratio_mean"] < 0.05
                            else "UNKNOWN"
                        )
                        print(f"  ✅ {variant}: std_ratio={diag['std_ratio_mean']:.4f} "
                              f"[{diag['std_ratio_min']:.4f}..{diag['std_ratio_max']:.4f}] "
                              f"KS_mean={diag['ks_p_mean']:.4f} "
                              f"NaN={diag['nan_count']} → {verdict}")
                        break
                    except Exception as e:
                        print(f"  ⚠️ {variant}: diagnostic error — {e}")

        if len(completed) >= len(job_ids) and len(results) >= len(variants):
            print("\n=== All jobs done, all artifacts diagnosed ===")
            break

        if len(results) >= len(completed) and len(active) == 0:
            print("\n=== No more active jobs ===")
            break

        time.sleep(args.poll_interval)

    # Final verdict table
    print("\n" + "=" * 72)
    print("FINAL VERDICT TABLE")
    print("=" * 72)
    print(f"{'Variant':<28} {'StdRatio':>8} {'Min':>8} {'KS_p':>8} {'Verdict':<12}")
    print("-" * 72)
    for variant in variants:
        if variant in results:
            r = results[variant]
            sr = r["std_ratio_mean"]
            verdict = (
                "✅ HEALTHY" if 0.7 <= sr <= 1.5
                else "⚠️ LOW" if sr < 0.7
                else "⚠️ HIGH" if sr > 1.5
                else "🔴 DEAD"
            )
            print(f"{variant:<28} {sr:>8.4f} {r['std_ratio_min']:>8.4f} "
                  f"{r['ks_p_mean']:>8.4f} {verdict:<12}")
        else:
            print(f"{variant:<28} {'—':>8} {'—':>8} {'—':>8} {'⏳ PENDING':<12}")

    # Save results
    import json
    results_path = output_root / "diagnostics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
