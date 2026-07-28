"""Conditional time-series diffusion adapter.

Thin subclass of :class:`UnconditionalTSDiffusionAdapter` that swaps the
underlying model class from ``TSDiff`` (unconditional) to ``TSDiffCond``
(conditional, mask-aware variant).  All fit/generate logic is inherited
from the unconditional adapter; we only override the import path and the
per-channel model construction kwargs.

The vendored package ``unconditional-time-series-diffusion`` ships both
classes in the same module tree:

    TSDiff      -> uncond_ts_diff.model.diffusion.tsdiff.TSDiff
    TSDiffCond  -> uncond_ts_diff.model.diffusion.tsdiff_cond.TSDiffCond

This adapter reuses the unconditional adapter's sys.path + gluonts shim
infrastructure (single source of truth for vendored-package bootstrap).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from src.experiments.adapters.deep_learning.utsd_adapter import UnconditionalTSDiffusionAdapter


def _ensure_gluonts_shim() -> None:
    """Idempotent shim for the gluonts modules ``tsdiff_cond.py`` imports.

    The unconditional adapter already shims ``gluonts.torch.modules.scaler``
    (used by TSDiff).  TSDiffCond additionally needs PyTorchPredictor +
    lagged_sequence_values from ``gluonts.torch.{model.predictor, util}``.
    Re-using whatever the corresponding modules expose keeps the symbol
    lookup at module-load time working without dragging in the full
    gluonts package.
    """
    if "gluonts.torch.modules.scaler" not in sys.modules:
        _install_gluonts_modules_scaler_shim()

    for alias in ("gluonts.torch.model.predictor", "gluonts.torch.model"):
        if alias in sys.modules:
            continue
        try:
            importlib.import_module(alias)
        except Exception:
            # Provide a stub package whose sub-modules import lazily.
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


import sys  # noqa: E402  (after _install_gluonts_modules_scaler_shim def that captures it)


class ConditionalTSDiffusionAdapter(UnconditionalTSDiffusionAdapter):
    """Conditional-diffusion sibling of UnconditionalTSDiffusionAdapter."""

    model_name = "ConditionalTSDiffusion"

    @staticmethod
    def _import_utsd():  # noqa: D401  (inherits semantics)
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

        # TSDiffCond lives in its own module (not in tsdiff.py).
        tsdiff_cond = importlib.import_module(
            "uncond_ts_diff.model.diffusion.tsdiff_cond"
        )
        return diffusion_configs, tsdiff_cond.TSDiffCond

    def fit(self, fit_input, checkpoints_dir, logs_dir):  # type: ignore[override]
        """Override only the per-channel model construction kwargs.

        Differences vs the unconditional flavour:
            * Class swap: ``TSDiff`` -> ``TSDiffCond`` (via ``_import_utsd``).
            * Extra kwarg: ``noise_observed=False`` (standard train_cond_model
              default; the conditional model has a noise-observed branch that
              we keep OFF because our data is fully observed).
        """
        # Inline-mirror the unconditional fit loop but swap the model class
        # and constructor kwargs.  Kept here (not delegated) to avoid coupling
        # the unconditional adapter's contract to a conditional kwarg that's
        # spurious for ``TSDiff``.
        import copy
        from typing import List

        import torch

        from src.experiments.adapters.deep_learning.training_utils import (
            EarlyStopping,
            FitTrainingInfo,
            make_loader,
            parse_training_params,
            resolve_device,
        )

        fit_input_batch = fit_input.batch
        windows = fit_input_batch.train_windows
        valid_windows = fit_input_batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError(
                "ConditionalTSDiffusionAdapter expects train_windows shaped (N, L, C)"
            )
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError(
                "ConditionalTSDiffusionAdapter requires non-empty valid_windows for model selection."
            )

        diffusion_configs, TSDiffCond = self._import_utsd()
        params = parse_training_params(fit_input)
        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])
        self.context_length = max(2, self.base_length // 2)
        self.prediction_length = max(1, self.base_length - self.context_length)
        self.models = []
        self.checkpoints = []

        device = resolve_device(fit_input.device)
        channel_val_losses: List[float] = []
        stopped_early = False
        best_epochs: List[int] = []

        for c in range(self.num_channels):
            cfg = copy.deepcopy(diffusion_configs.diffusion_small_config)
            model = TSDiffCond(
                **cfg,
                freq="h",
                use_features=False,
                use_lags=False,
                normalization="none",
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                lr=params.learning_rate,
                init_skip=True,
                noise_observed=False,
            ).to(device)

            channel_train = windows[:, :, c].unsqueeze(-1).float()
            channel_valid = valid_windows[:, :, c].unsqueeze(-1).float()
            train_loader = make_loader(channel_train, params.batch_size, shuffle=True)
            valid_loader = make_loader(channel_valid, params.batch_size, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            early_stop = EarlyStopping(patience=params.patience)
            best_state = copy.deepcopy(model.state_dict())
            info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

            for epoch in range(params.max_epochs):
                model.train()
                train_loss = 0.0
                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    t = torch.randint(0, model.timesteps, (batch_x.shape[0],), device=device).long()
                    loss, _, _ = model.p_losses(batch_x, t, features=None, loss_type="l2")
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item())
                info.train_loss_history.append(train_loss / max(len(train_loader), 1))

                val_loss = self._eval_val_loss(model, valid_loader, device)
                info.val_loss_history.append(val_loss)
                if val_loss < info.best_val_loss:
                    info.best_val_loss = val_loss
                    info.best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                if early_stop.step(val_loss, epoch + 1):
                    info.stopped_early = True
                    break

            model.load_state_dict(best_state)
            channel_val_losses.append(info.best_val_loss)
            best_epochs.append(info.best_epoch)
            stopped_early = stopped_early or info.stopped_early
            self.models.append(model)

        self._is_fitted = True
        # Persist ONE consolidated FINAL checkpoint (multi-channel dict)
        # labeled with the seq length so downstream regeneration loads one
        # file. Per-channel ctsd_checkpoint_{c+1}.pt stays for in-process
        # generation; the consolidated file is the canonical handoff.
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": len(self.models),
                "base_length": self.base_length,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "channels": [
                    {"channel": c, "state_dict": m.state_dict()}
                    for c, m in enumerate(self.models)
                ],
            },
            final_ckpt,
        )
        # Track the consolidated ckpt as the sole model checkpoint.
        self.checkpoints = [final_ckpt]
        return {
            "num_channels": self.num_channels,
            "best_val_loss": float(sum(channel_val_losses) / len(channel_val_losses)),
            "best_epoch": int(max(best_epochs)),
            "stopped_early": stopped_early,
        }
