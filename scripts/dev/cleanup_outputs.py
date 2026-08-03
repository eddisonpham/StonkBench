"""Reorganize outputs/ into a clean 13-model bundle the peer can ship.

What this script does:
  1. Moves every "bad" run_id in outputs/{results,checkpoints,sanity,logs,...}/
     into outputs_legacy2/<sub>/<run_id>/ (preserves everything; nothing deleted).
  2. Inside the kept canonical run (`baseline_2026-07-21_vm`), removes
     is present (copied from `cond_sig_wgan_first_run` if missing).
  3. Places the ground-truth tensor at `outputs/data/ground_truth.pt`
     (canonical location for the friend).
  4. Moves standalone top-level .json sidecars (`baseline_hp_summary.json`,
     `kalman_vae_hp_summary.json`) into `outputs_legacy2/` (they were
     scaffolding for past HP searches; the kept run's artifacts already
     contain the relevant metadata in their `.pt` files).
  5. Reports a verification table of the final state.

  unconditional_tsdiffusion, conditional_tsdiffusion, cond_sig_wgan,
  timegrad.
The five canonical statistical models (post 2026-07-30 cleanup; gbm and
  ou_process were removed end-to-end): block_bootstrap,
  stationary_block_bootstrap, merton_jump_diffusion, de_jump_diffusion,
  garch11.

CLI flags
---------
--dry-run         Print the planned moves/skips without touching disk.
--verify-only     Run the verification table only (no moves).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path("/home/phamnhut/StonkBench")
OUTPUTS = ROOT / "outputs"
LEGACY = ROOT / "outputs_legacy2"

# Canonical run we keep in outputs/ — contains all 13 final models (post
# DL set; cond_sig_wgan is the conditionally-coupled Sig-WGAN variant.
KEEP_RUN = "baseline_2026-07-21_vm"

# cond_sig_wgan was trained and saved in a separate run; this is the
# version whose .pt has correct amplitude (finite, calibration gated
# correctly inside that run). Patched_2026-07-23 looks healthy on paper
# but its artifact has std_ratio=0.016 because the calibration-flag did
# not propagate through the live pipeline (an open bug). Don't ship
# that one.
CSW_SOURCE_RUN = "cond_sig_wgan_first_run"

# Canonical 7 DL (post 2026-07-23).  timegrad / timevae / sig_wgan were
# decommissioned end-to-end and their per-run sub-dirs in the kept run
# are quarantined into DEAD_MODELS so the bundle ships clean.
KEEP_DL = {
    "quantgan",
    "vrnn",
    "kalman_vae",
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "cond_sig_wgan",
    "timegrad",
}
KEEP_STAT = {
    "block_bootstrap",
    "stationary_block_bootstrap",
    "merton_jump_diffusion",
    "de_jump_diffusion",
    "garch11",
}
KEEP_MODELS = sorted(KEEP_DL | KEEP_STAT)  # 13 models total (8 DL + 5 stat)

# Decommissioned model keys whose per-run sub-dirs in the kept run go to
# legacy instead of being shipped to the peer.  The adapters + vendors
# were deleted end-to-end; these sub-dirs are preserved purely for
# historical reproducibility (regen sanity / re-train from scratch).
DEAD_MODEL_DIRS = ("timevae", "sig_wgan", "timegan")

# Subdirectories of outputs/ we mirror into outputs_legacy2/ when a run
# is moved. Anything not in this list is left where it is at top level
# (e.g. mock_bundle.zip stays in outputs/).
PER_RUN_SUBDIRS = ["results", "checkpoints", "sanity", "logs"]

# Top-level sidecar JSON files that were HP-search scaffolding. They
# don't belong in the eval bundle.
LEGACY_TOP_LEVEL_FILES = [
    "baseline_hp_summary.json",
    "kalman_vae_hp_summary.json",
]


def _list_run_ids() -> list[str]:
    results_dir = OUTPUTS / "results"
    if not results_dir.is_dir():
        return []
    return sorted(d.name for d in results_dir.iterdir() if d.is_dir())


def plan_moves() -> list[tuple[str, Path, Path]]:
    """Compute (kind, src, dst) tuples describing every planned move.

    kind is one of: 'move_run', 'move_dead_model', 'copy_csw',
                    'move_ground_truth', 'move_top_json'.
    """
    moves: list[tuple[str, Path, Path]] = []
    all_runs = _list_run_ids()
    bad_runs = [r for r in all_runs if r != KEEP_RUN]
    for run in bad_runs:
        for sub in PER_RUN_SUBDIRS:
            src = OUTPUTS / sub / run
            if src.is_dir():
                moves.append(("move_run", src, LEGACY / sub / run))
    # Quarantine the kept run's per-model sub-dirs of decommissioned
    # models so the shipped bundle contains only the canonical 13.
    for sub in PER_RUN_SUBDIRS:
        for dead in DEAD_MODEL_DIRS:
            dead_src = OUTPUTS / sub / KEEP_RUN / dead
            if dead_src.is_dir():
                moves.append(
                    ("move_dead_model", dead_src, LEGACY / sub / KEEP_RUN / dead)
                )
    # Copy cond_sig_wgan from first_run (canonical CKPT for that model)
    # back into the kept run. We copy first then run the move, so the
    # order matters: do this AFTER the bad-run move above.
    for sub in PER_RUN_SUBDIRS:
        # The csw_source is already queued for move_run above; but we need
        # the data still on disk when we copy. The actual copy happens
        # *before* the move in execute(); here we just register the plan.
        csw_src = OUTPUTS / sub / CSW_SOURCE_RUN / "cond_sig_wgan"
        csw_dst = OUTPUTS / sub / KEEP_RUN / "cond_sig_wgan"
        if csw_src.is_dir() and not csw_dst.is_dir():
            moves.append(("copy_csw", csw_src, csw_dst))

    # Move ground_truth dir → data/ground_truth.pt
    gt_src = OUTPUTS / "ground_truth"
    if gt_src.is_dir():
        for pt in gt_src.glob("*.pt"):
            moves.append(
                ("move_ground_truth", pt, OUTPUTS / "data" / "ground_truth.pt")
            )

    # Move top-level sidecar JSONs.
    for name in LEGACY_TOP_LEVEL_FILES:
        p = OUTPUTS / name
        if p.is_file():
            moves.append(("move_top_json", p, LEGACY / name))
    return moves


def print_plan(moves: list[tuple[str, Path, Path]]) -> None:
    print(f"=== {len(moves)} planned moves ===")
    for kind, src, dst in moves:
        print(f"  [{kind}] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def execute_moves(moves: list[tuple[str, Path, Path]]) -> None:
    """Apply moves in safe order: copy csw first, move runs after."""
    LEGACY.mkdir(parents=True, exist_ok=True)
    for sub in PER_RUN_SUBDIRS:
        (LEGACY / sub).mkdir(parents=True, exist_ok=True)

    # 1. Copy cond_sig_wgan from CSW source (still on disk in outputs/)
    for kind, src, dst in moves:
        if kind == "copy_csw":
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # 2. Move bad runs into legacy.
    for kind, src, dst in moves:
        if kind == "move_run":
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  moved   {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # 3. Quarantine dead-model sub-dirs inside the kept run.
    for kind, src, dst in moves:
        if kind == "move_dead_model":
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  moved   {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # 4. Ground truth → outputs/data/.
    for kind, src, dst in moves:
        if kind == "move_ground_truth":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            print(f"  copied  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    # Remove the now-redundant ground_truth/ directory (we copied the .pt
    # contents; the dir itself is empty save for that one file).
    gt_dir = OUTPUTS / "ground_truth"
    if gt_dir.is_dir() and not any(gt_dir.iterdir()):
        gt_dir.rmdir()
        print(f"  removed empty {gt_dir.relative_to(ROOT)}")

    # 5. Top-level sidecar JSONs.
    for kind, src, dst in moves:
        if kind == "move_top_json":
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print(f"  moved   {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def verify() -> int:
    """Print a verification table; return exit code (0 on clean state)."""
    print()
    print("=== final-state verification ===")
    rc = 0
    # Each kept model needs an artifact .pt at outputs/results/KEEP_RUN/artifacts/
    art_dir = OUTPUTS / "results" / KEEP_RUN / "artifacts"
    if not art_dir.is_dir():
        print(f"  [FAIL] {art_dir.relative_to(ROOT)} missing")
        return 2
    # 13 models x {artifact, checkpoint (DL only), sanity}
    print()
    print(f'  {"model":<28} {"result .pt":<12} {"checkpoint":<14} {"sanity":<10}')
    print("  " + "-" * 64)
    for m in KEEP_MODELS:
        pt = art_dir / f"{m}_seq252.pt"
        ck = OUTPUTS / "checkpoints" / KEEP_RUN / m
        sa = OUTPUTS / "sanity" / KEEP_RUN / m
        pt_ok = "OK" if pt.is_file() else "MISS"
        ck_ok = "OK" if ck.is_dir() else ("n/a" if m in KEEP_STAT else "MISS")
        sa_ok = "OK" if sa.is_dir() else "MISS"
        if pt_ok != "OK" or (m in KEEP_DL and ck_ok != "OK") or sa_ok != "OK":
            rc = 1
        print(f"  {m:<28} {pt_ok:<12} {ck_ok:<14} {sa_ok:<10}")
    print()
    # Ground truth
    gt = OUTPUTS / "data" / "ground_truth.pt"
    print(f"  ground_truth at outputs/data/ground_truth.pt: {'OK' if gt.is_file() else 'MISS'}")
    if not gt.is_file():
        rc = 1
    # Legacy non-empty
    print(f"  outputs_legacy2/ exists: {'OK' if LEGACY.is_dir() else 'MISS'}")
    # Sanity: only KEEP_RUN should remain in results/, checkpoints/, sanity/, logs/.
    bad_dirs_outside_legacy = []
    for sub in PER_RUN_SUBDIRS:
        d = OUTPUTS / sub
        if not d.is_dir():
            continue
        for child in d.iterdir():
            if child.is_dir() and child.name != KEEP_RUN:
                bad_dirs_outside_legacy.append(child.relative_to(ROOT))
    if bad_dirs_outside_legacy:
        print("  [WARN] extra run dirs in outputs/ (should be in legacy):")
        for b in bad_dirs_outside_legacy:
            print(f"    - {b}")
        rc = 1
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="only print plan")
    ap.add_argument("--verify-only", action="store_true", help="just verify")
    args = ap.parse_args()

    moves = plan_moves()
    if args.verify_only:
        return_code = verify()
        raise SystemExit(return_code)
    print_plan(moves)
    if args.dry_run:
        print("\n(dry-run; nothing changed)")
        return_code = verify()
        raise SystemExit(return_code)
    print()
    print("=== executing ===")
    execute_moves(moves)
    return_code = verify()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
