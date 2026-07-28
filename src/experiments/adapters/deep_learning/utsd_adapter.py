"""Unconditional time-series diffusion adapter — native multivariate.

Trains a single TSDiff model on the full (N, L, C) multivariate input.
The backbone's ``input_init`` is a ``nn.Linear(input_dim, hidden_dim)`` that
processes each timestep independently across channels, so setting
``input_dim=C`` gives native multivariate denoising.  The S4 residual blocks
use ``Conv1d`` across the time axis, preserving cross-channel structure.

Key change vs the old per-channel adapter: ``sample_n(return_lags=True)``
returns the full ``(R, L, input_dim)`` tensor instead of the hardcoded
``samples[..., 0]`` slice.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
)
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput


class UnconditionalTSDiffusionAdapter(ModelAdapter):
    """Native-multivariate unconditional time-series diffusion adapter.

    Trains one diffusion model on the full (N, L, C) tensor.  The backbone
    accepts any ``input_dim`` so multivariate works out of the box.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "UnconditionalTSDiffusion"
        self.model: Any = None
        self.num_channels: int = 0
        self.base_length: int = 0
        self.context_length: int = 0
        self.prediction_length: int = 0
        self.checkpoints: List[Path] = []

    # ------------------------------------------------------------------
    # Vendor bootstrap (unchanged — sets up sys.path + gluonts shim)
    # ------------------------------------------------------------------
    @staticmethod
    def _import_utsd():
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

        if "gluonts.torch.modules.scaler" not in sys.modules:
            try:
                from gluonts.torch.scaler import MeanScaler, NOPScaler  # type: ignore

                scaler_mod = ModuleType("gluonts.torch.modules.scaler")
                scaler_mod.MeanScaler = MeanScaler
                scaler_mod.NOPScaler = NOPScaler
                sys.modules["gluonts.torch.modules.scaler"] = scaler_mod
            except Exception:
                pass

        import uncond_ts_diff.configs as diffusion_configs  # type: ignore

        model_pkg = ModuleType("uncond_ts_diff.model")
        model_pkg.__path__ = [str(model_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model"] = model_pkg
        diffusion_pkg = ModuleType("uncond_ts_diff.model.diffusion")
        diffusion_pkg.__path__ = [str(diffusion_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model.diffusion"] = diffusion_pkg

        tsdiff_module = importlib.import_module(
            "uncond_ts_diff.model.diffusion.tsdiff"
        )
        return diffusion_configs, tsdiff_module.TSDiff

    # ------------------------------------------------------------------
    # Model construction hook — subclasses override for TSDiffCond etc.
    # ------------------------------------------------------------------
    def _build_model(
        self,
        backbone_params: Dict[str, Any],
        cfg: Dict[str, Any],
        context_length: int,
        prediction_length: int,
        lr: float,
        device: torch.device,
    ) -> Any:
        """Build the diffusion model.  Subclasses may override to swap
        the model class (e.g. TSDiffCond) or add kwargs."""
        diffusion_configs, TSDiff = self._import_utsd()
        return TSDiff(
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
        ).to(device)

    # ------------------------------------------------------------------
    # Validation loss (operates on full multivariate batch)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_val_loss(
        self, model: Any, loader: DataLoader, device: torch.device
    ) -> float:
        model.eval()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)  # (B, L, C)
            t = torch.randint(
                0, model.timesteps, (batch_x.shape[0],), device=device
            ).long()
            loss, _, _ = model.p_losses(batch_x, t, features=None, loss_type="l2")
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # fit — single joint model on (N, L, C)
    # ------------------------------------------------------------------
    def fit(
        self,
        fit_input: AdapterFitInput,
        checkpoints_dir: Path,
        logs_dir: Path,
    ) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError(
                "UnconditionalTSDiffusionAdapter expects train_windows shaped (N, L, C)"
            )
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError(
                "UnconditionalTSDiffusionAdapter requires non-empty valid_windows."
            )

        diffusion_configs, _TSDiff_cls = self._import_utsd()
        params = parse_training_params(fit_input)

        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])
        self.context_length = max(2, self.base_length // 2)
        self.prediction_length = max(1, self.base_length - self.context_length)

        device = resolve_device(fit_input.device)

        # --- Build single joint model with input_dim=C ---
        cfg = copy.deepcopy(diffusion_configs.diffusion_small_config)
        backbone_params = cfg["backbone_parameters"].copy()
        backbone_params["input_dim"] = self.num_channels
        backbone_params["output_dim"] = self.num_channels

        model = self._build_model(
            backbone_params=backbone_params,
            cfg=cfg,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            lr=params.learning_rate,
            device=device,
        )

        # --- Data loaders (full multivariate, no per-channel split) ---
        train_loader = make_loader(
            windows.float(), params.batch_size, shuffle=True
        )
        valid_loader = make_loader(
            valid_windows.float(), params.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        early_stop = EarlyStopping(patience=params.patience)
        best_state = copy.deepcopy(model.state_dict())
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
                best_state = copy.deepcopy(model.state_dict())
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
    # generate — sample_n(return_lags=True) gives full (R, L, C)
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, generation_length: int, seed: int
    ) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        with torch.no_grad():
            # return_lags=True → full (R, seq_len, input_dim) array
            sampled = self.model.sample_n(
                num_samples=num_samples, return_lags=True
            )

        data = torch.from_numpy(sampled).float()  # (R, seq_len, C)
        # Trim to requested generation_length
        L = min(generation_length, data.shape[1])
        data = data[:, :L, :]

        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "utsd_multivariate"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
