"""Conditional time-series diffusion adapter — native multivariate.

Thin subclass of :class:`UnconditionalTSDiffusionAdapter` that swaps the
underlying model class from ``TSDiff`` (unconditional) to ``TSDiffCond``
(conditional, mask-aware variant).  All fit/generate logic is inherited
from the unconditional adapter; we only override the import path, the
per-channel model construction kwargs, and the generation path.

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

    Uses ``TSDiffCond`` for the model class.  Inherits the native-multivariate
    ``fit()`` from the parent (single joint model on (N, L, C)).  Overrides
    ``_import_utsd`` and ``generate`` to use the conditional model's
    ``forecast()`` method.
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

    # ------------------------------------------------------------------
    # fit — parent handles single-joint model; we add noise_observed kwarg
    # ------------------------------------------------------------------
    def fit(self, fit_input, checkpoints_dir, logs_dir):  # type: ignore[override]
        """Override to pass ``noise_observed=False`` to TSDiffCond."""
        # The parent's fit() builds the model via TSDiff(**cfg, ...).
        # We need TSDiffCond with noise_observed=False instead.
        # Solution: temporarily monkey-patch _import_utsd's TSDiff class,
        # then delegate to parent, then restore.
        #
        # Simpler: just inline the model construction here and call the
        # parent's training loop via super().fit() won't work because
        # the parent creates TSDiff. So we override fit() completely,
        # duplicating the parent logic with TSDiffCond-specific kwargs.
        import copy as _copy

        from src.experiments.adapters.deep_learning.training_utils import (
            EarlyStopping,
            FitTrainingInfo,
            make_loader,
            parse_training_params,
            resolve_device,
        )

        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError(
                "ConditionalTSDiffusionAdapter expects train_windows shaped (N, L, C)"
            )
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError(
                "ConditionalTSDiffusionAdapter requires non-empty valid_windows."
            )

        diffusion_configs, TSDiffCond = self._import_utsd()
        params = parse_training_params(fit_input)

        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])
        self.context_length = max(2, self.base_length // 2)
        self.prediction_length = max(1, self.base_length - self.context_length)

        device = resolve_device(fit_input.device)

        # --- Build single joint TSDiffCond with input_dim=C ---
        cfg = _copy.deepcopy(diffusion_configs.diffusion_small_config)
        backbone_params = cfg["backbone_parameters"].copy()
        backbone_params["input_dim"] = self.num_channels
        backbone_params["output_dim"] = self.num_channels

        model = TSDiffCond(
            backbone_parameters=backbone_params,
            timesteps=cfg["timesteps"],
            diffusion_scheduler=cfg["diffusion_scheduler"],
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

        # --- Data loaders (full multivariate) ---
        train_loader = make_loader(
            windows.float(), params.batch_size, shuffle=True
        )
        valid_loader = make_loader(
            valid_windows.float(), params.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        early_stop = EarlyStopping(patience=params.patience)
        best_state = _copy.deepcopy(model.state_dict())
        info = FitTrainingInfo(
            best_val_loss=float("inf"), best_epoch=0, stopped_early=False
        )

        for epoch in range(params.max_epochs):
            model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                t = torch.randint(
                    0, model.timesteps, (batch_x.shape[0],), device=device
                ).long()
                loss, _, _ = model.p_losses(
                    batch_x, t, features=None, loss_type="l2"
                )
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
            info.train_loss_history.append(train_loss / max(len(train_loader), 1))

            val_loss = self._eval_val_loss(model, valid_loader, device)
            info.val_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = _copy.deepcopy(model.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        model.load_state_dict(best_state)
        self.model = model

        # --- Persist single checkpoint ---
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": self.num_channels,
                "base_length": self.base_length,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "input_dim": self.num_channels,
                "state_dict": model.state_dict(),
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]
        self._is_fitted = True

        return {
            "num_channels": self.num_channels,
            "best_val_loss": info.best_val_loss,
            "best_epoch": info.best_epoch,
            "stopped_early": info.stopped_early,
        }

    # ------------------------------------------------------------------
    # generate — use forecast() for full (B, L, C) output
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, generation_length: int, seed: int
    ):
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
