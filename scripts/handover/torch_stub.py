"""Minimal ``torch`` stand-in so numpy-only code paths can be exercised without torch.

``src.taxonomies`` and friends import torch at module level, but the P&L and
portfolio evaluators only touch numpy.  Installing this stub lets the bundle
tooling (``recompute_downstream.py``, ``audit_downstream.py``) and the unit
tests import those modules in an environment that has no PyTorch.

Any real torch call raises, so a code path that actually needs torch fails
loudly instead of silently returning something wrong::

    from torch_stub import install_torch_stub
    install_torch_stub()          # no-op if real torch is already imported
"""

from __future__ import annotations

import sys


class _Unavailable:
    """Callable placeholder that raises when the real torch would have run."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("torch is stubbed out; only numpy code paths are available")

    def __getattr__(self, name):  # pragma: no cover - attribute access on a stub
        raise RuntimeError(f"torch is stubbed out (tried to read torch.{name})")


class _Tensor:
    """Type marker so ``isinstance(x, torch.Tensor)`` checks keep working."""

    pass


class TorchStub:
    """Drop-in module object: ``torch.Tensor`` exists, everything else raises."""

    __version__ = "stub"
    Tensor = _Tensor

    def __getattr__(self, name):  # pragma: no cover - reached only on real use
        return _Unavailable()


def install_torch_stub() -> bool:
    """Register the stub in ``sys.modules`` unless real torch is importable.

    Returns ``True`` if the stub was installed, ``False`` if real torch is
    already present (so callers can branch if they care).
    """
    if "torch" in sys.modules:
        return not isinstance(sys.modules["torch"], TorchStub)
    try:
        import torch  # noqa: F401  (real torch is available; leave it alone)
    except Exception:
        sys.modules["torch"] = TorchStub()
        return True
    return False
