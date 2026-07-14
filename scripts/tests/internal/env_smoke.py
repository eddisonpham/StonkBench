"""Phase 1 import + behavior smoke for ``src/utils/env.py`` and the swept entry points.

Run from the repo root::

    PYTHONPATH=. python scripts/tests/internal/env_smoke.py

Stages:

* Stage A — env-only helpers, no torch required.
* Stage B — full module imports (with torch).
* Stage C — ``--help`` dry runs on swept entry points (with torch).

This is a one-shot Phase-1 verification script. Delete after Phase 1 ends.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


# scripts/tests/internal/ -> scripts/tests/ -> scripts/ -> REPO
REPO = Path(__file__).resolve().parents[3]
assert REPO.name == "StonkBench", f"unexpected repo root layout: {REPO}"


def _try_import(mods):
    failures = []
    for m in mods:
        try:
            importlib.import_module(m)
            print(f"  OK  {m}")
        except Exception as exc:  # noqa: BLE001
            failures.append((m, exc))
            print(f"  ERR {m}: {exc.__class__.__name__}: {exc}")
    return failures


def env_only_check():
    print("=== Stage A: env.py behavior (torch-free) ===")
    from src.utils.env import (
        DEFAULT_OUTPUT_ROOT_NAME,
        apply_run_id_override,
        get_dl_set_path,
        get_local_jobs,
        get_output_root,
        get_run_id,
        get_stats_set_path,
        get_visible_cuda_indices,
        is_smoke,
        project_root,
        set_run_id,
    )

    print(f"project_root() = {project_root()}")
    print(f"get_output_root() = {get_output_root()}")
    print(f"get_dl_set_path() = {get_dl_set_path()}")
    print(f"get_stats_set_path() = {get_stats_set_path()}")
    print(f"DEFAULT_OUTPUT_ROOT_NAME = {DEFAULT_OUTPUT_ROOT_NAME!r}")
    print(f"get_run_id() = {get_run_id()}")

    # is_smoke flag
    os.environ["STONKBENCH_SMOKE"] = "1"
    assert is_smoke() is True
    assert is_smoke(False) is False
    assert is_smoke(True) is True
    print("is_smoke() ok (True/False/None)")
    os.environ.pop("STONKBENCH_SMOKE", None)

    # Mock cpu_count so clamp assertions are deterministic regardless of host
    with patch("os.cpu_count", return_value=8):
        os.environ.pop("STONKBENCH_LOCAL_JOBS", None)
        assert get_local_jobs() == 1, "default 1 when unset"
        assert get_local_jobs(3) == 3, "explicit default respected"
        os.environ["STONKBENCH_LOCAL_JOBS"] = "999"
        assert get_local_jobs() == 8, "clamped to cpu_count mock"
        os.environ["STONKBENCH_LOCAL_JOBS"] = "4"
        assert get_local_jobs() == 4, "within cpu_count: no clamp"
        os.environ["STONKBENCH_LOCAL_JOBS"] = "0"
        assert get_local_jobs() == 1, "zero clamps to 1"
        os.environ["STONKBENCH_LOCAL_JOBS"] = "-5"
        assert get_local_jobs() == 1, "negative clamps to 1"
        os.environ["STONKBENCH_LOCAL_JOBS"] = "abc"
        assert get_local_jobs() == 1, "garbage falls back to default(1)"
        os.environ.pop("STONKBENCH_LOCAL_JOBS", None)
    print("get_local_jobs() clamp & edge-cases OK (mocked cpu_count=8)")

    # Output root precedence
    expected = str(get_output_root())
    argparse_default = os.environ.get("STONKBENCH_OUTPUT_ROOT") or str(get_output_root())
    assert argparse_default == expected
    os.environ["STONKBENCH_OUTPUT_ROOT"] = "/tmp/phase1_test_root"
    argparse_after_env = os.environ.get("STONKBENCH_OUTPUT_ROOT") or str(get_output_root())
    assert argparse_after_env == "/tmp/phase1_test_root", "env precedence honored"
    os.environ.pop("STONKBENCH_OUTPUT_ROOT", None)
    print("output_root argparse/env precedence OK")

    # visible_cuda_indices
    os.environ["STONKBENCH_CUDA_VISIBLE_DEVICES"] = "0,3"
    indices = get_visible_cuda_indices()
    assert indices == [0, 3], f"expected [0, 3], got {indices}"
    print(f"STONKBENCH_CUDA_VISIBLE_DEVICES=0,3 -> indices={indices}")
    os.environ.pop("STONKBENCH_CUDA_VISIBLE_DEVICES", None)

    # Run-ID round-trip
    chosen = apply_run_id_override("phase1_smoke")
    assert chosen == "phase1_smoke"
    assert os.environ["STONKBENCH_RUN_ID"] == "phase1_smoke"
    assert get_run_id() == "phase1_smoke"
    print(f"run_id round-trip OK -> {chosen}")
    os.environ.pop("STONKBENCH_RUN_ID", None)

    # Run-ID date fallback (snapshot once to avoid midnight flake)
    expected_date = datetime.now().date().isoformat()
    os.environ.pop("STONKBENCH_RUN_ID", None)
    assert get_run_id() == expected_date, f"expected {expected_date}, got {get_run_id()}"
    print(f"run_id date fallback OK -> {expected_date}")


TORCH_MODULES = [
    "src.utils.env",
    "src.utils.device",
    "src.utils.preprocessed_data_utils",
    "src.utils.artifact_utils",
    "src.experiments.core.contracts",
    "src.experiments.core.registry",
    "src.experiments.core.io",
    "src.experiments.core.pipeline",
    "src.experiments.hp_configs",
    "src.experiments.hp_search",
    "src.experiments.run_benchmark",
    "src.experiments.run_final_training",
]


def full_import_check():
    print("\n=== Stage B: full import (requires torch) ===")
    failures = _try_import(TORCH_MODULES)
    if not failures:
        from src.experiments.core.io import resolve_run_id as io_resolve_run_id
        from src.utils.env import get_run_id as env_get_run_id
        assert io_resolve_run_id is env_get_run_id, (
            "core/io.resolve_run_id must be the SAME function object as env.get_run_id"
        )
        print("  OK  identity core.io.resolve_run_id is env.get_run_id")
        from src.utils.preprocessed_data_utils import resolve_dl_set_path as pdu_resolve_dl
        from src.utils.env import get_dl_set_path as env_get_dl
        dl_value = pdu_resolve_dl()
        dl_env_value = str(env_get_dl())
        assert dl_value == dl_env_value, (
            f"delegation mismatch: preprocessed_data_utils={dl_value!r}, env={dl_env_value!r}"
        )
        print(f"  OK  delegation match: resolve_dl_set_path() == {dl_value}")
    return failures


def help_check():
    print("\n=== Stage C: --help dry runs ===")
    env = {**os.environ, "PYTHONPATH": str(REPO), "STONKBENCH_RUN_ID": "smoke-help"}
    for entry in [
        "-m src.experiments.run_benchmark",
        "-m src.experiments.run_final_training",
        "-m src.experiments.hp_search",
    ]:
        try:
            r = subprocess.run(
                [sys.executable, entry, "--help"],
                capture_output=True,
                text=True,
                env=env,
                timeout=20,
            )
            ok = r.returncode == 0
            print(f"  {'OK ' if ok else 'ERR'} {entry} (exit={r.returncode})")
            if ok:
                for line in r.stdout.splitlines():
                    if "output_root" in line:
                        print(f"    {line.strip()[:240]}")
            else:
                print(f"    stderr head: {r.stderr[:300]}")
        except FileNotFoundError as exc:
            print(f"  ERR {entry}: interpreter missing ({exc})")
        except subprocess.TimeoutExpired:
            print(f"  ERR {entry}: timeout")


def main():
    env_only_check()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\nTorch not installed; skipping Stage B + Stage C for this host.")
        return
    failures = full_import_check()
    help_check()
    if failures:
        sys.exit(1)
    print("\nALL ENV CHECKS PASSED")


if __name__ == "__main__":
    main()
