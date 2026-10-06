#!/usr/bin/env python
"""Rebuild <run_dir>/report.md + plots/all_curves.png from partial outputs.

Safe to run while jobs are still executing — reads whatever is on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_logger import refresh_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_dir",
        required=True,
        help="Mini run dir, e.g. /scratch/epham/stonkbench/mini_exp/<run_id>",
    )
    args = p.parse_args()
    report = refresh_report(Path(args.run_dir))
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
