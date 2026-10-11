"""Shared test bootstrap.

Imported at the top of every test module so the suite runs the same way under
``python3 -m unittest discover -s tests`` and ``pytest tests/``, in an
environment that may not have PyTorch installed::

    from _bootstrap import bootstrap
    bootstrap()
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLING = REPO_ROOT / "scripts" / "handover"

_HAS_REAL_TORCH: bool | None = None


def bootstrap() -> bool:
    """Put the repo and the bundle tooling on ``sys.path``, install the torch stub.

    Returns ``True`` when real torch is available, ``False`` when the stub is in
    use (so a test that genuinely needs torch can skip with a clear reason).
    Safe to call from every module; the work happens once.
    """
    global _HAS_REAL_TORCH
    for path in (REPO_ROOT, TOOLING):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if _HAS_REAL_TORCH is None:
        from torch_stub import install_torch_stub

        _HAS_REAL_TORCH = not install_torch_stub()
    return _HAS_REAL_TORCH
