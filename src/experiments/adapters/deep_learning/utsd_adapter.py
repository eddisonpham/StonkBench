from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List
from types import ModuleType

import torch
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
)
from src.utils.artifact_utils import stitch_sequences


class UnconditionalTSDiffusionAdapter(ModelAdapter):
    """
    Real unconditional time-series diffusion integration using repository model code.

    Trains one diffusion model per channel and stacks outputs to (R, L, C).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "UnconditionalTSDiffusion"
        self.models: List = []
        self.base_length = 1
        self.num_channels = 1
        self.context_length = 1
        self.prediction_length = 1
        self.checkpoints: List[Path] = []

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

        tsdiff_module = importlib.import_module("uncond_ts_diff.model.diffusion.tsdiff")
        TSDiff = tsdiff_module.TSDiff

        return diffusion_configs, TSDiff

    @torch.no_grad()
    def _eval_val_loss(self, model, loader: DataLoader, device: torch.device) -> float:
        model.eval()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            t = torch.randint(0, model.timesteps, (batch_x.shape[0],), device=device).long()
            loss, _, _ = model.p_losses(batch_x, t, features=None, loss_type="l2")
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("UnconditionalTSDiffusionAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("UnconditionalTSDiffusionAdapter requires non-empty valid_windows for model selection.")

        diffusion_configs, TSDiff = self._import_utsd()
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
            model = TSDiff(
                **cfg,
                freq="H",
                use_features=False,
                use_lags=False,
                normalization="none",
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                lr=params.learning_rate,
                init_skip=True,
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
            ckpt = checkpoints_dir / f"utsd_checkpoint_{c + 1}.pt"
            torch.save(model.state_dict(), ckpt)
            self.checkpoints.append(ckpt)

        self._is_fitted = True
        return {
            "num_channels": self.num_channels,
            "best_val_loss": float(sum(channel_val_losses) / len(channel_val_losses)),
            "best_epoch": int(max(best_epochs)),
            "stopped_early": stopped_early,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        per_channel = []
        for c, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                sampled = model.sample_n(num_samples=num_samples, return_lags=False)
            channel = torch.from_numpy(sampled).float()
            if generation_length != channel.shape[1]:
                channel = stitch_sequences(channel, generation_length, seed=seed + c)
            per_channel.append(channel.unsqueeze(-1))

        data = torch.cat(per_channel, dim=-1)
        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "utsd_real"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
