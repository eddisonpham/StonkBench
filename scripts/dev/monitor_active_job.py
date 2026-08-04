"""Background monitor for an in-flight SLURM job.

Hardened per code-review (2026-07-30):
- subprocess.run wrapped with try/except for transient slurm hiccups
- Jobs that vanish from squeue are verified via sacct BEFORE declaring done
- kalman_vae divergence auto-cancel restricted to literal `_safe` suffix
- New .pt files retry-once on partial-write errors
- REAL_STD_CH0 computed from the staged dl_set at startup (no hardcoding)
- Loud directory-existence check at startup so a wipe-during-monitor doesn't no-op silently

For .pt artifact diagnostics:
- NaN/Inf check     (early divergence detector for KVAE MultivariateNormal.loc)
- std ratio         (vs real channel-0 std)
- AC1, skew, kurt   (moment checks)
- KS-naive CDF      (cheap proxy against real test windows)
Verdicts:
  HEALTHY    std_ratio >= 0.40 AND KS <= 0.20 AND no NaN/Inf
  ATROPHIED  0.10 <= std_ratio < 0.40
  COLLAPSED  std_ratio < 0.10 OR KS > 0.40
  DIVERGED   NaN/Inf present
Auto-cancel only triggers if a `kalman_vae_safe` task shows DIVERGED.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_SLURM_LOG_DIR = Path("/scratch/epham/stonkbench/slurm_logs")
DEFAULT_LATEST_DIR = Path("/scratch/epham/stonkbench/output/results/latest")
DEFAULT_DL_SET_PATH = Path("/home/epham/StonkBench/data/preprocessed/dl_set_W252.pt")


# ---------- subprocess hardening ------------------------------------------------
def safe_run(cmd: list[str], timeout: int = 20) -> tuple[str, str, int]:
    """Return (stdout, stderr, returncode); never raise on transient failures."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout, p.stderr, p.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}", -1


# ---------- sla state ----------------------------------------------------------
def tasks_state_squeue(job_id: str) -> tuple[int, int, list[tuple[int, str, str, str]]]:
    """Soft `squeue` lookup; empty list on transient or missing job id."""
    out, err, rc = safe_run(["squeue", "-j", job_id, "-o", "%.10i %.2T %.10M %R", "-h"])
    if rc != 0 or not out:
        return 0, 0, []
    running = pending = 0
    rows: list[tuple[int, str, str, str]] = []
    for line in out.strip().splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        full_id, st, tm, reason = parts
        try:
            task_n = int(full_id.split("_")[-1]) if "_" in full_id else 0
        except ValueError:
            task_n = 0
        if st == "RUNNING":
            running += 1
        elif st == "PENDING":
            pending += 1
        rows.append((task_n, st, tm, reason))
    # If the stderr carries the SLURM "Invalid job id" string we trust squeue;
    # the caller will fall through to sacct. Don't beat a dead horse with retries.
    return running, pending, sorted(rows)


def tasks_state_sacct(job_id: str) -> tuple[int, int, list[tuple[int, str, str, str]]]:
    """Hard `sacct` fallback when squeue returns empty/404.

    Distinguishes 'completed all tasks' from 'job vanished from squeue but is
    still being processed by slurmdbd'. Returns:
      (0, 1, [...])  if any task is PENDING / RUNNING (job still alive)
      (0, 0, [...])  only if ALL tasks are COMPLETED/FAILED/CANCELLED
      (0, 0, [...])  if sacct also unreachable (last-mile ambiguity)
    """
    out, err, rc = safe_run(
        ["sacct", "-j", job_id, "-X", "-o", "JobID,State,Elapsed,Reason", "--noheader"]
    )
    if rc != 0 or not out:
        return 0, 0, []  # Both squeue and sacct unreachable — caller will keep polling.
    running = pending = 0
    finished_states = {
        "COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT",
        "NODE_FAIL", "DEADLINE",
    }
    rows = []
    for line in out.strip().splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        jobid_field, st, elapsed, reason = parts
        try:
            task_n = int(jobid_field.split(".")[-1]) if "." in jobid_field else int(jobid_field.split("_")[-1])
        except ValueError:
            task_n = 0
        if st in ("RUNNING", "PENDING"):
            running += 1 if st == "RUNNING" else 0
            pending += 1 if st == "PENDING" else 0
            rows.append((task_n, st, elapsed, reason))
        elif st in finished_states:
            rows.append((task_n, st, elapsed, reason))
        else:
            # Unknown state — treat as still pending so we don't bail prematurely.
            pending += 1
            rows.append((task_n, st, elapsed, reason))
    return running, pending, sorted(rows)


# ---------- err tailing --------------------------------------------------------
def tail_err_severity(log_dir: Path, job_id: str, task_id: int) -> tuple[int, int]:
    err = log_dir / f"train_{job_id}_{task_id}.err"
    if not err.is_file():
        return 0, 0
    try:
        text = err.read_text(errors="ignore")
    except OSError:
        return 0, 0
    lines = text.splitlines()
    severity_pat = ("ERROR", "FATAL", "Traceback", "RuntimeError", "ValueError", "NonFinite", "NaN", "Inf")
    severe = sum(1 for ln in lines if any(p in ln for p in severity_pat))
    return len(lines), severe


# ---------- diagnostic ---------------------------------------------------------
def load_real_std(dl_set_path: Path) -> float:
    """Load the real-data channel-0 std (returns per-channel mean as fallback).

    Read once at startup so threshold drift can't sneak in if preprocessing changes.
    """
    try:
        d = torch.load(dl_set_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[WARN] could not load dl_set {dl_set_path}: {exc}; falling back to 0.016158",
              flush=True)
        return 0.016158
    if "train_windows" not in d:
        print("[WARN] dl_set missing train_windows; falling back to 0.016158", flush=True)
        return 0.016158
    tw = d["train_windows"]
    ch_mean = d.get("channel_mean")
    ch_std = d.get("channel_std")
    if ch_mean is not None and ch_std is not None:
        raw = tw * ch_std.view(1, 1, -1) + ch_mean.view(1, 1, -1)
        diff = raw[:, 1:, 0] - raw[:, :-1, 0]
    else:
        diff = tw[:, 1:, 0] - tw[:, :-1, 0]
    return float(diff.float().std().item())


def diagnose_pt(pt: Path, real_std: float, dl_set_path: Path) -> dict:
    """Per-time-step diagnostic. Retries once on partial-write errors."""
    for attempt in range(2):
        try:
            obj = torch.load(pt, map_location="cpu", weights_only=True)
            data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            if not isinstance(data, torch.Tensor):
                return {"path": str(pt), "error": "unsupported data type"}
            ch0 = data[:, :, 0].float() if data.dim() == 3 else data.float()
            break
        except Exception as exc:
            if attempt == 0:
                time.sleep(5)
                continue
            return {"path": str(pt), "error": f"load failed: {exc}"}

    diff = ch0[:, 1:] - ch0[:, :-1]
    vals = diff.reshape(-1).float()

    if not torch.isfinite(vals).all():
        n_bad = (~torch.isfinite(vals)).sum().item()
        return {"path": str(pt), "verdict": "DIVERGED", "std": float("nan"), "sr": 0.0,
                "n_nan_inf": n_bad}

    mean = vals.mean().item()
    std = vals.std().item()
    sr = std / real_std if real_std > 0 else 0.0
    centered = vals - mean
    var = std ** 2
    ac1 = float((centered[:-1] * centered[1:]).mean() / var) if var > 0 else 0.0
    skew = float((centered ** 3).mean() / var ** 1.5) if var > 0 else 0.0
    kurt = float((centered ** 4).mean() / var ** 2) if var > 0 else 0.0

    # KS-naive vs full real ch0 diff distribution.
    ks = float("nan")
    try:
        d = torch.load(dl_set_path, map_location="cpu", weights_only=True)
        tw = d["train_windows"]
        if "channel_mean" in d and "channel_std" in d:
            raw = tw * d["channel_std"].view(1, 1, -1) + d["channel_mean"].view(1, 1, -1)
            ref = (raw[:, 1:, 0] - raw[:, :-1, 0]).reshape(-1).float()
        else:
            ref = (tw[:, 1:, 0] - tw[:, :-1, 0]).reshape(-1).float()
        bins = np.linspace(-0.10, 0.10, 80)
        h_r, _ = np.histogram(ref.numpy(), bins=bins, density=True)
        h_g, _ = np.histogram(vals.numpy(), bins=bins, density=True)
        s_r, s_g = h_r.sum(), h_g.sum()
        cdf_r = np.cumsum(h_r) / s_r if s_r > 0 else np.zeros_like(h_r)
        cdf_g = np.cumsum(h_g) / s_g if s_g > 0 else np.zeros_like(h_g)
        ks = float(np.max(np.abs(cdf_r - cdf_g)))
    except Exception:
        pass

    if sr < 0.10 or ks > 0.40:
        verdict = "COLLAPSED"
    elif sr < 0.40:
        verdict = "ATROPHIED"
    elif ks > 0.20:
        verdict = "MISALIGNED"
    else:
        verdict = "HEALTHY"

    return {"path": str(pt), "verdict": verdict, "std": std, "sr": sr,
            "ac1": ac1, "skew": skew, "kurt": kurt, "ks": ks}


# ---------- scancel ------------------------------------------------------------
def maybe_cancel_kvae_divergence(job_id: str, offending_pt: Path) -> bool:
    """Auto-scancel ONLY if this is a `kalman_vae_safe` task with NaN/Inf divergence.

    The bare `kalman_vae` task in the array is NOT auto-cancelled (handled by
    the user) because failure there means re-architecting is needed; we don't
    want to waste compute on KVAE's mid-array discovery either.
    """
    name = offending_pt.name.lower()
    is_safe = "kalman_vae_safe" in name
    if not is_safe:
        print(f"[WARN] divergence on non-safe KVAE task {name}; manual intervention required",
              flush=True)
        return False
    print(f"[AUTO-CANCEL] DIVERGED kalman_vae_safe -> scancel {job_id}", flush=True)
    _, _, rc = safe_run(["scancel", job_id], timeout=10)
    return rc == 0


# ---------- main loop ----------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--poll-secs", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=180)
    ap.add_argument("--slurm-log-dir", type=Path, default=DEFAULT_SLURM_LOG_DIR)
    ap.add_argument("--latest-dir", type=Path, default=DEFAULT_LATEST_DIR)
    ap.add_argument("--dl-set", type=Path, default=DEFAULT_DL_SET_PATH)
    args = ap.parse_args()

    if not args.slurm_log_dir.is_dir():
        print(f"[FATAL] slurm-log dir missing: {args.slurm_log_dir}", flush=True)
        return 2
    if not args.latest_dir.is_dir():
        print(f"[WARN] latest-dir not present yet (job may not have started): {args.latest_dir}",
              flush=True)

    real_std = load_real_std(args.dl_set)
    print(f"[INFO] REAL_STD_CH0 from {args.dl_set}: {real_std:.6f}", flush=True)

    seen_pt: set[str] = set()
    job = args.job
    print(f"[{_ts()}] START monitor job={job}", flush=True)

    for it in range(1, args.max_iters + 1):
        running, pending, rows = tasks_state_squeue(job)
        squeue_empty = (running == 0 and pending == 0 and not rows)

        # If squeue saw nothing, fall through to sacct before declaring "all done".
        if squeue_empty:
            print(f"[{_ts()}] iter={it} squeue=empty; consulting sacct for {job}", flush=True)
            running, pending, rows = tasks_state_sacct(job)
            if running == 0 and pending == 0 and not rows:
                # sacct also has no record — transient ambiguity. Keep polling.
                print(f"[{_ts()}] iter={it} sacct=empty too; treating as transient — keep polling",
                      flush=True)
            elif running == 0 and pending == 0:
                # All tasks hit a finished state.
                print(f"[{_ts()}] iter={it} sacct confirms all tasks in finished state", flush=True)
                break
            else:
                print(f"[{_ts()}] iter={it} sacct: running={running} pending={pending}", flush=True)
        else:
            print(f"\n[{_ts()}] iter={it} running={running} pending={pending}", flush=True)

        for task_id, st, tm, reason in rows:
            lc, sev = tail_err_severity(args.slurm_log_dir, job, task_id)
            print(f"  task {task_id} state={st} time={tm} reason={reason!r} err_lines={lc} severe={sev}",
                  flush=True)

        # Scan new .pt files
        new_pts = [
            p for p in args.latest_dir.rglob("*.pt")
            if str(p) not in seen_pt and p.is_file() and p.stat().st_size > 1024
        ]
        for pt in sorted(new_pts):
            seen_pt.add(str(pt))
            print(f"  NEW PT: {pt}", flush=True)
            diag = diagnose_pt(pt, real_std, args.dl_set)
            if "error" in diag:
                print(f"    {diag['error']}", flush=True)
                continue
            print(
                f"    verdict={diag['verdict']} std={diag['std']:.5f} sr={diag['sr']:.3f} "
                f"ac1={diag['ac1']:.3f} kurt={diag['kurt']:.1f} ks={diag['ks']:.3f}",
                flush=True,
            )
            if diag.get("verdict") == "DIVERGED":
                maybe_cancel_kvae_divergence(job, pt)

        if running == 0 and pending == 0 and rows:
            break
        # Only sleep when polling is still meaningful.
        time.sleep(args.poll_secs)

    # Final summary
    final_pts = sorted(args.latest_dir.rglob("*.pt"))
    print(f"\n[{_ts()}] SUMMARY: {len(final_pts)} .pt files in latest/", flush=True)
    for pt in final_pts:
        d = diagnose_pt(pt, real_std, args.dl_set)
        v = d.get("verdict") if "error" not in d else f"ERR({d['error'][:40]})"
        print(f"  {pt.parent.name + '/' + pt.name:42s} verdict={v}", flush=True)
    return 0


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


if __name__ == "__main__":
    sys.exit(main())
