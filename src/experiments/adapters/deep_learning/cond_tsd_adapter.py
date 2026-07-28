"""Conditional time-series diffusion adapter — native multivariate.

Thin subclass of :class:`UnconditionalTSDiffusionAdapter` that swaps the
underlying model class from ``TSDiff`` (unconditional) to ``TSDiffCond``
(conditional, mask-aware variant).  All fit/generate logic is inherited
from the unconditional adapter; we only override the import path, the
model construction hook (``_build_model``), and the generation path.

Key multivariate change: ``forecast(obs, mask)`` returns the full
``output`` tensor instead of the hardcoded ``forward()`` → ``pred[..., 0]``.
The adapter creates an all-zeros observation + mask so the model denoises
from pure noise (unconditional generation through the conditional model).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import torch

from src.experiments.core.contracts import AdapterGenerateOutput
from src.experiments.adapters.deep_learning.utsd_adapter import (
    UnconditionalTSDiffusionAdapter,
)


def _ensure_gluonts_shim() -> None:
    """Idempotent shim for the gluonts modules ``tsdiff_cond.py`` imports."""
    if "gluonts.torch.modules.scaler" not in sys.modules:
        _install_gluonts_modules_scaler_shim()

    for alias in ("gluonts.torch.model.predictor", "gluonts.torch.model"):
        if alias in sys.modules:
            continue
        try:
            importlib.import_module(alias)
        except Exception:
            stub = ModuleType(alias)
            stub.__path__ = []  # type: ignore[attr-defined]
            sys.modules[alias] = stub

    if "gluonts.torch.util" not in sys.modules:
        try:
            importlib.import_module("gluonts.torch.util")
        except Exception:
            sys.modules["gluonts.torch.util"] = ModuleType("gluonts.torch.util")


def _install_gluonts_modules_scaler_shim() -> None:
    try:
        from gluonts.torch.scaler import MeanScaler, NOPScaler  # type: ignore

        mod = ModuleType("gluonts.torch.modules.scaler")
        mod.MeanScaler = MeanScaler
        mod.NOPScaler = NOPScaler
        sys.modules["gluonts.torch.modules.scaler"] = mod
    except Exception:
        pass


class ConditionalTSDiffusionAdapter(UnconditionalTSDiffusionAdapter):
    """Conditional-diffusion sibling of UnconditionalTSDiffusionAdapter.

    Uses ``TSDiffCond`` for the model class.  Inherits ``fit()`` entirely
    from the parent — only ``_import_utsd``, ``_build_model``, and
    ``generate`` are overridden.
    """

    model_name = "ConditionalTSDiffusion"

    @staticmethod
    def _import_utsd():  # noqa: D401
        """Import TSDiffCond instead of TSDiff from the same vendored tree."""
        _ensure_gluonts_shim()
        root = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "deep_learning"
            / "unconditional-time-series-diffusion"
            / "src"
        )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        pkg_root = root / "uncond_ts_diff"
        model_root = pkg_root / "model"
        diffusion_root = model_root / "diffusion"

        import uncond_ts_diff.configs as diffusion_configs  # type: ignore

        model_pkg = ModuleType("uncond_ts_diff.model")
        model_pkg.__path__ = [str(model_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model"] = model_pkg
        diffusion_pkg = ModuleType("uncond_ts_diff.model.diffusion")
        diffusion_pkg.__path__ = [str(diffusion_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model.diffusion"] = diffusion_pkg

        tsdiff_cond = importlib.import_module(
            "uncond_ts_diff.model.diffusion.tsdiff_cond"
        )
        return diffusion_configs, tsdiff_cond.TSDiffCond

    def _build_model(
        self,
        backbone_params: Dict[str, Any],
        cfg: Dict[str, Any],
        context_length: int,
        prediction_length: int,
        lr: float,
        device: torch.device,
    ) -> Any:
        """Build TSDiffCond instead of TSDiff."""
        diffusion_configs, TSDiffCond = self._import_utsd()
        return TSDiffCond(
            backbone_parameters=backbone_params,
            timesteps=cfg["timesteps"],
            diffusion_scheduler=cfg["diffusion_scheduler"],
            freq="h",
            use_features=False,
            use_lags=False,
            normalization="none",
            context_length=context_length,
            prediction_length=prediction_length,
            lr=lr,
            init_skip=True,
            noise_observed=False,
        ).to(device)

    # ------------------------------------------------------------------
    # generate — use forecast() for full (B, L, C) output
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, generation_length: int, seed: int
    ) -> AdapterGenerateOutput:
        """Unconditional generation via TSDiffCond.forecast().

        We pass zeros as observation with an all-zeros mask so the model
        denoises from pure noise (all positions are "unobserved").
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        seq_len = self.context_length + self.prediction_length

        with torch.no_grad():
            observation = torch.zeros(
                num_samples, seq_len, self.num_channels,
                device=next(self.model.parameters()).device,
            )
            # All-zeros mask = all positions unobserved → unconditional gen
            observation_mask = torch.zeros_like(observation)
            sampled = self.model.forecast(
                observation=observation,
                observation_mask=observation_mask,
                features=None,
            )

        data = sampled.cpu().float()  # (B, seq_len, C)
        L = min(generation_length, data.shape[1])
        data = data[:, :L, :]

        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "ctsd_multivariate"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
