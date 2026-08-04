#!/usr/bin/env python3
"""GAN collapse monitor for Wave-3 architecture-fix recovery (2026-07-30).

artifacts/ for new .pt artifacts. When one lands (or its mtime changes since
last seen), runs a per-channel collapse diagnostic against the train reference:

- Train reference is auto-selected:
    1) /scratch/epham/stonkbench/output/ground_truth/ground_truth_seq{N}.pt (matches the
       z-scored files used by sanity plots)
    2) Fallback to /home/epham/StonkBench/data/preprocessed/dl_set_W252.pt
- Per-channel metrics: std_ratio, KS, AC1, skew, kurtosis.
- Verdict cascades (matches scripts/dev/diagnose_collapse.py exactly):
    SEVERE COLLAPSE  mean std_ratio < 0.05
    COLLAPSED        < 0.20
    ATROPHIED        < 0.50
    UNDER-DISPERSED  < 0.80
    HEALTHY          [0.80, 1.30]
    OVER-DISPERSED   > 1.30
    DIVERGED         NaN / Inf in sim data (replaces HEALTHY; loud)

Auto-cancel policy: NEVER cancel on SEVERE / COLLAPSED / ATROPHIED
diagnostic verdicts (collapse is expected during GAN recovery).  Auto-cancel
ONLY for NaN/Inf DIVERGED — same as monitor_active_job.py.

Output:
- stdout: human-readable per-event log with verdict + per-channel summary
- /scratch/epham/stonkbench/slurm_logs/gan_collapse_results.jsonl: append-only
  one JSON object per detection (path, mtime, model_key, seq_length, ts,
  per-channel ratios, mean_ratio, verdict, error).

Run:
    python scripts/dev/gan_collapse_monitor.py [--poll-secs 60] [--latest-dir ...]
                                              [--results-jsonl ...] [--once]

The script assumes /home/epham/.venvs/stonkbench/bin/python (project venv)
but is stdlib + numpy + torch + scipy only at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp, skew, kurtosis

DEFAULT_LATEST_DIR = Path("/scratch/epham/stonkbench/output/results/latest")
DEFAULT_GROUND_TRUTH_DIR = Path("/scratch/epham/stonkbench/output/ground_truth")
DEFAULT_DL_SET_PATH = Path("/home/epham/StonkBench/data/preprocessed/dl_set_W252.pt")
DEFAULT_RESULTS_JSONL = Path("/scratch/epham/stonkbench/slurm_logs/gan_collapse_results.jsonl")
DEFAULT_LOG_PATH = Path("/scratch/epham/stonkbench/slurm_logs/gan_collapse_monitor.log")

WATCHED_SEQ_LENGTHS = (21, 42, 126, 252)

# Model-key prefixes this monitor watches. `discover_pts` prefix-matches each
# latest/<subdir> name, so both the bare name (cond_sig_wgan) and any variant
# (cond_sig_wgan_qtrain5_h100) are picked up without allowlist edits.
# (pcf_gan was deleted 2026-07-30 per user directive — intentionally absent.)
WATCHED_MODELS = ("cond_sig_wgan", "quantgan")

# ---- Verdict thresholds (match diagnose_collapse.py cascading) --------------
def verdict_from_ratio(ratio: float, ks_d: float) -> str:
    """Cascading SEVERE → COLLAPSED → ... → HEALTHY/OVER-DISPERSED verdict."""
    if not np.isfinite(ratio):
        return "DIVERGED"
    if ratio < 0.05 or ks_d > 0.50:
        return "SEVERE COLLAPSE"
    if ratio < 0.20:
        return "COLLAPSED"
    if ratio < 0.50:
        return "ATROPHIED"
    if ratio < 0.80:
        return "UNDER-DISPERSED"
    if ratio > 1.30:
        return "OVER-DISPERSED"
    return "HEALTHY"


# ---- Result dataclass -------------------------------------------------------
@dataclass
class CollapseResult:
    path: str
    mtime: float
    ts: str
    model_key: str
    seq_length: int
    train_ref_path: str
    verdict: str
    mean_std_ratio: float
    n_healthy: int           # channels with ratio in [0.7, 1.3]
    n_compressed: int
    n_collapsed: int         # ratio < 0.40 (collapse-flavoured)
    n_nan_inf: int
    per_channel_ratios: list[float] = field(default_factory=list)
    per_channel_ks: list[float] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, default=str)


# ---- Train reference loaders ------------------------------------------------
def _load_ground_truth(gt_dir: Path, seq_length: int) -> np.ndarray | None:
    """Try ground_truth_seq{N}.pt (matches sanity plot overlays)."""
    pt = gt_dir / f"ground_truth_seq{seq_length}.pt"
    if not pt.is_file():
        return None
    try:
        d = torch.load(pt, map_location="cpu", weights_only=False)
        data = d["data"] if isinstance(d, dict) and "data" in d else d
        if not isinstance(data, torch.Tensor):
            return None
        return data.float().numpy()
    except Exception as exc:
        print(f"[WARN] failed to load {pt}: {exc}", flush=True)
        return None


def _load_dl_set_fallback(dl_path: Path) -> np.ndarray | None:
    """Fallback to z-scored dl_set_W252.pt train_windows."""
    try:
        d = torch.load(dl_path, map_location="cpu", weights_only=True)
        if "train_windows" not in d:
            return None
        return d["train_windows"].float().numpy()
    except Exception as exc:
        print(f"[WARN] failed to load {dl_path}: {exc}", flush=True)
        return None


# ---- Per-channel diagnostic -------------------------------------------------
def diagnose(sim_pt: Path, gt_dir: Path, dl_set_path: Path, seq_length: int,
             subsample_n: int = 1000) -> CollapseResult:
    """One-shot diagnostic on a single sim .pt."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    mtime = sim_pt.stat().st_mtime
    model_key = sim_pt.parent.parent.name
    base = {
        "path": str(sim_pt),
        "mtime": mtime,
        "ts": ts,
        "model_key": model_key,
        "seq_length": seq_length,
    }

    try:
        # ------- Load sim data -------
        obj = torch.load(sim_pt, map_location="cpu", weights_only=False)
        data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        if not isinstance(data, torch.Tensor):
            return CollapseResult(
                **base, train_ref_path="", verdict="ERR",
                mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
                n_nan_inf=0, error="sim data is not a tensor",
            )
        sim = data.float()
        if sim.dim() == 3:
            # (N, L, C)
            sim_np = sim.numpy()
        elif sim.dim() == 2:
            # (N, C) — flatten to single seq slot
            sim_np = sim.unsqueeze(1).numpy()
        else:
            return CollapseResult(
                **base, train_ref_path="", verdict="ERR",
                mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
                n_nan_inf=0, error=f"unsupported shape {tuple(sim.shape)}",
            )

        # ------- NaN / Inf check -------
        if not np.isfinite(sim_np).all():
            n_bad = int((~np.isfinite(sim_np)).sum())
            return CollapseResult(
                **base, train_ref_path="", verdict="DIVERGED",
                mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
                n_nan_inf=n_bad, error=f"non-finite values in sim (n={n_bad})",
            )

        # ------- Pick train reference -------
        ref_np = _load_ground_truth(gt_dir, seq_length)
        ref_kind = "ground_truth"
        ref_path = str(gt_dir / f"ground_truth_seq{seq_length}.pt") if ref_np is not None else ""
        if ref_np is None:
            ref_np = _load_dl_set_fallback(dl_set_path)
            ref_kind = "dl_set"
            ref_path = str(dl_set_path) if ref_np is not None else ""
        if ref_np is None:
            return CollapseResult(
                **base, train_ref_path="", verdict="ERR",
                mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
                n_nan_inf=0, error="no train reference available",
            )

        # ------- Align (subsample sim down + use as many ref as needed) -------
        # Per-channel stats only need flat arrays per channel; reshape sim to
        # (samples * time, channels) and similarly for reference.
        N, L, C = sim_np.shape
        sim_flat = sim_np.reshape(-1, C).astype(np.float64)
        ref_flat = ref_np.reshape(-1, ref_np.shape[-1]).astype(np.float64)
        if C > ref_flat.shape[-1]:
            return CollapseResult(
                **base, train_ref_path=ref_path, verdict="ERR",
                mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
                n_nan_inf=0, error=f"sim has {C} channels > ref {ref_flat.shape[-1]}",
            )
        # take first C channels of reference if it has extra
        ref_flat = ref_flat[:, :C]

        # Subsample for KS speed
        n_sub = min(subsample_n, sim_flat.shape[0])
        rng = np.random.default_rng(42)
        sidx = rng.choice(sim_flat.shape[0], n_sub, replace=False) if sim_flat.shape[0] > n_sub else slice(None)
        ridx = rng.choice(ref_flat.shape[0], n_sub, replace=False) if ref_flat.shape[0] > n_sub else slice(None)
        if isinstance(sidx, slice):
            sim_sub = sim_flat
        else:
            sim_sub = sim_flat[sidx]
        if isinstance(ridx, slice):
            ref_sub = ref_flat
        else:
            ref_sub = ref_flat[ridx]

        # ------- Per-channel metrics -------
        per_chan_ratio = []
        per_chan_ks = []
        for ch in range(C):
            s = sim_flat[:, ch]
            r = ref_flat[:, ch]
            s_std = float(np.std(s))
            r_std = float(np.std(r))
            ratio = s_std / r_std if r_std > 1e-12 else 0.0
            ks_p, ks_d = ks_2samp(r[ridx if not isinstance(ridx, slice) else slice(None)],
                                   s[sidx if not isinstance(sidx, slice) else slice(None)])
            per_chan_ratio.append(ratio)
            per_chan_ks.append(ks_d)

        ratios_arr = np.array(per_chan_ratio)
        ks_arr = np.array(per_chan_ks)
        mean_ratio = float(ratios_arr.mean())
        mean_ks = float(ks_arr.mean())

        # Ac1 / skew / kurt across whole sim for context (pooled)
        sim_pooled = sim_np.reshape(-1)
        pooled_ac1 = float(np.corrcoef(sim_pooled[:-1], sim_pooled[1:])[0, 1])
        pooled_sk = float(skew(sim_pooled))
        pooled_ku = float(kurtosis(sim_pooled, fisher=True))

        # ------- Verdict (cascading thresholds + KS penalty) -------
        # Knock DOWN to SEVERE if mean KS is very high (< 0.50 threshold already
        # embedded into verdict_from_ratio via the ks_d arg). Better: use mean ks.
        verdict = verdict_from_ratio(mean_ratio, mean_ks)

        n_healthy = int(np.sum((ratios_arr >= 0.7) & (ratios_arr <= 1.3)))
        n_compressed = int(np.sum((ratios_arr >= 0.4) & (ratios_arr < 0.7)))
        n_collapsed = int(np.sum(ratios_arr < 0.4))

        return CollapseResult(
            **base,
            train_ref_path=f"{ref_kind}:{ref_path}",
            verdict=verdict,
            mean_std_ratio=mean_ratio,
            n_healthy=n_healthy,
            n_compressed=n_compressed,
            n_collapsed=n_collapsed,
            n_nan_inf=0,
            per_channel_ratios=per_chan_ratio,
            per_channel_ks=per_chan_ks,
            error=None,
        )
    except Exception as exc:
        return CollapseResult(
            **base, train_ref_path="", verdict="ERR",
            mean_std_ratio=0.0, n_healthy=0, n_compressed=0, n_collapsed=0,
            n_nan_inf=0, error=f"{type(exc).__name__}: {exc}",
        )


# ---- File discovery ---------------------------------------------------------
def discover_pts(latest_dir: Path,
                 models: tuple[str, ...] = WATCHED_MODELS) -> list[tuple[Path, int]]:
    """Find every (sim_path, seq_length) under watched model subdirs.

    Match logic: every immediate subdir of ``latest_dir`` whose name STARTS
    with one of the watched model prefixes. This catches both legacy
    ``cond_sig_wgan_mc500_d2``, etc., without needing explicit allowlist edits
    every time a new variant is wired.
    """
    out: list[tuple[Path, int]] = []
    if not latest_dir.is_dir():
        return out
    for sub in sorted(latest_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not any(sub.name == m or sub.name.startswith(m + "_") or sub.name.startswith(m + "-")
                   for m in models):
            continue
        art_dir = sub / "artifacts"
        if not art_dir.is_dir():
            continue
        for pt in sorted(art_dir.glob("*.pt")):
            for seq in WATCHED_SEQ_LENGTHS:
                if f"_seq{seq}.pt" in pt.name:
                    out.append((pt, seq))
                    break
    return out


# ---- Logging ----------------------------------------------------------------
def _log_line(log_path: Path, line: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        print(f"[WARN] log write failed: {exc}", flush=True)


def _append_jsonl(jsonl_path: Path, result: CollapseResult) -> None:
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a") as fh:
            fh.write(result.to_json() + "\n")
    except Exception as exc:
        print(f"[WARN] jsonl write failed: {exc}", flush=True)


# ---- Signal handling --------------------------------------------------------
def _install_signal_handlers() -> dict[str, int]:
    """Capture SIGINT/SIGTERM as soft-stop (set internal flag)."""
    state = {"stop": 0}

    def _handler(signum, _frame):
        state["stop"] = 1
        print(f"\n[SIGNAL] received {signum}; draining after current iteration",
              flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return state


# ---- Main loop --------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest-dir", type=Path, default=DEFAULT_LATEST_DIR)
    ap.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    ap.add_argument("--dl-set", type=Path, default=DEFAULT_DL_SET_PATH)
    ap.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    ap.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    ap.add_argument("--poll-secs", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=0,
                    help="0 = run until SIGINT/SIGTERM (default). Set positive for finite runs.")
    ap.add_argument("--once", action="store_true",
                    help="Single-shot: discover all current .pt files, diagnose each, exit.")
    ap.add_argument("--models", type=str, default=",".join(WATCHED_MODELS),
                    help="Comma-separated subset of models to watch (default: all watched).")
    args = ap.parse_args()

    sig_state = _install_signal_handlers()
    watched_models: tuple[str, ...] = tuple(
        m.strip() for m in args.models.split(",") if m.strip()
    )

    print(f"[INIT] GAN collapse monitor", flush=True)
    print(f"[INIT] watched_models = {watched_models}", flush=True)
    print(f"[INIT] latest_dir  = {args.latest_dir}", flush=True)
    print(f"[INIT] gt_dir      = {args.ground_truth_dir}", flush=True)
    print(f"[INIT] dl_set      = {args.dl_set}", flush=True)
    print(f"[INIT] jsonl       = {args.results_jsonl}", flush=True)
    print(f"[INIT] log_path    = {args.log_path}", flush=True)
    print(f"[INIT] poll_secs   = {args.poll_secs}", flush=True)
    print(f"[INIT] once        = {args.once}", flush=True)

    seen: dict[str, float] = {}  # path -> last mtime we processed
    it = 0
    while True:
        it += 1
        cur_pts = discover_pts(args.latest_dir, watched_models)
        if not cur_pts:
            if it == 1:
                print(f"[INFO] no GAN artifacts under {args.latest_dir} yet; "
                      "waiting for first .pt to land", flush=True)
            if args.once:
                print("[ONCE] no artifacts to process; exiting", flush=True)
                return 0
            if args.max_iters and it >= args.max_iters:
                print(f"[EXIT] max_iters={args.max_iters} reached", flush=True)
                return 0
            time.sleep(args.poll_secs)
            if sig_state["stop"]:
                print("[EXIT] signal received", flush=True)
                return 0
            continue

        new_or_changed = []
        for pt, seq in cur_pts:
            key = str(pt)
            mt = pt.stat().st_mtime
            if key not in seen or seen[key] != mt:
                new_or_changed.append((pt, seq, mt))

        if not new_or_changed:
            if not args.once:
                if args.max_iters and it >= args.max_iters:
                    print(f"[EXIT] max_iters={args.max_iters} reached", flush=True)
                    return 0
            time.sleep(args.poll_secs)
            if args.once:
                print("[ONCE] all current .pt files already processed; exiting", flush=True)
                return 0
            if sig_state["stop"]:
                print("[EXIT] signal received", flush=True)
                return 0
            continue

        for pt, seq, mt in sorted(new_or_changed, key=lambda t: t[2]):
            if sig_state["stop"]:
                print("[SIGNAL] stop mid-iteration; reverting seen flag for incomplete run",
                      flush=True)
                break
            result = diagnose(pt, args.ground_truth_dir, args.dl_set, seq)
            seen[str(pt)] = mt
            _append_jsonl(args.results_jsonl, result)

            line = (
                f"[{result.ts}] {pt.parent.parent.name}/{pt.name:<25s} "
                f"verdict={result.verdict:<18s} mean_ratio={result.mean_std_ratio:.3f} "
                f"healthy={result.n_healthy:>2d} compressed={result.n_compressed:>2d} "
                f"collapsed={result.n_collapsed:>2d} ref={result.train_ref_path}"
            )
            if result.error:
                line += f"  ERROR({result.error})"
            print(line, flush=True)
            _log_line(args.log_path, line)

            if result.verdict == "DIVERGED":
                print(f"[ACTION] DIVERGED on {pt}; manual review required (no auto-cancel "
                      "from this monitor — only this script's policy would be to NOT cancel "
                      "GAN collapse verdicts)", flush=True)

        if args.once:
            print("[ONCE] all current .pt files processed; exiting", flush=True)
            return 0
        if args.max_iters and it >= args.max_iters:
            print(f"[EXIT] max_iters={args.max_iters} reached", flush=True)
            return 0
        if sig_state["stop"]:
            print("[EXIT] signal received", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())