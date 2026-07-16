from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.calibration import ChannelMomentStats, match_channel_moments
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
    use_calibration,
)
class TimeGradAdapter(ModelAdapter):
    """Real TimeGrad network integration."""

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "TimeGrad"
        self.train_net = None
        self.pred_net = None
        self.base_length = 1
        self.num_channels = 1
        self.context_length = 1
        self.prediction_length = 1
        self.history_length = 2
        self.time_feat_dim = 1
        self.checkpoint: Path | None = None
        self.channel_stats: ChannelMomentStats | None = None
        self.past_target_template: torch.Tensor | None = None
        self.apply_calibration = False

    @staticmethod
    def _import_timegrad_models():
        # parents[4] = repo root (NOT src/). Vendored timegrad package lives at
        # <repo>/models/deep_learning/timegrad, NOT under src/.
        root = Path(__file__).resolve().parents[4] / "models" / "deep_learning" / "timegrad"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        if "utils" in sys.modules:
            utils_mod = sys.modules["utils"]
            utils_file = str(getattr(utils_mod, "__file__", ""))
            if "TimeGAN-pytorch" in utils_file:
                del sys.modules["utils"]

        from time_grad_network import TimeGradPredictionNetwork, TimeGradTrainingNetwork  # type: ignore

        return TimeGradTrainingNetwork, TimeGradPredictionNetwork

    def _build_batch_inputs(self, series_batch: torch.Tensor):
        bsz, _, channels = series_batch.shape
        past_target = series_batch[:, : self.history_length, :]
        future_target = series_batch[
            :, self.history_length : self.history_length + self.prediction_length, :
        ]

        target_dimension_indicator = (
            torch.arange(channels, device=series_batch.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
            .contiguous()
        )
        past_time_feat = torch.zeros(
            bsz, self.history_length, self.time_feat_dim, device=series_batch.device
        )
        future_time_feat = torch.zeros(
            bsz, self.prediction_length, self.time_feat_dim, device=series_batch.device
        )
        past_observed = torch.ones_like(past_target)
        future_observed = torch.ones_like(future_target)
        past_is_pad = torch.zeros(bsz, self.history_length, device=series_batch.device)
        return (
            target_dimension_indicator,
            past_time_feat,
            past_target,
            past_observed,
            past_is_pad,
            future_time_feat,
            future_target,
            future_observed,
        )

    def _forward_loss(self, batch_windows: torch.Tensor) -> torch.Tensor:
        (
            target_dimension_indicator,
            past_time_feat,
            past_target,
            past_observed,
            past_is_pad,
            future_time_feat,
            future_target,
            future_observed,
        ) = self._build_batch_inputs(batch_windows)
        loss, _, _ = self.train_net(
            target_dimension_indicator=target_dimension_indicator,
            past_time_feat=past_time_feat,
            past_target_cdf=past_target,
            past_observed_values=past_observed,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target,
            future_observed_values=future_observed,
        )
        return loss

    @torch.no_grad()
    def _eval_val_loss(self, loader: DataLoader, device: torch.device) -> float:
        self.train_net.eval()
        total = 0.0
        count = 0
        for (batch_windows,) in loader:
            batch_windows = batch_windows.to(device)
            total += float(self._forward_loss(batch_windows).item())
            count += 1
        return total / max(count, 1)

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("TimeGradAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("TimeGradAdapter requires non-empty valid_windows for model selection.")

        TimeGradTrainingNetwork, TimeGradPredictionNetwork = self._import_timegrad_models()
        params = parse_training_params(fit_input)
        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])
        self.channel_stats = ChannelMomentStats.from_windows(windows)
        self.apply_calibration = use_calibration(fit_input)
        self.context_length = max(2, self.base_length // 2)
        self.history_length = self.context_length + 1
        self.prediction_length = max(1, self.base_length - self.history_length)
        if self.history_length + self.prediction_length > self.base_length:
            raise ValueError("Invalid TimeGrad split setup for training window length.")
        idx = int(fit_input.seed) % windows.shape[0]
        self.past_target_template = windows[idx, : self.history_length, :].clone()

        input_size = len([1]) * self.num_channels + self.num_channels * 1 + self.time_feat_dim
        device = resolve_device(fit_input.device)
        self.train_net = TimeGradTrainingNetwork(
            input_size=input_size,
            num_layers=1,
            num_cells=max(16, self.num_channels * 8),
            cell_type="LSTM",
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=0.1,
            lags_seq=[1],
            target_dim=self.num_channels,
            conditioning_length=max(16, self.num_channels * 8),
            diff_steps=20,
            loss_type="l2",
            beta_end=0.1,
            beta_schedule="linear",
            residual_layers=4,
            residual_channels=max(8, self.num_channels * 4),
            dilation_cycle_length=2,
            scaling=True,
        ).to(device)

        train_loader = make_loader(windows, params.batch_size, shuffle=True)
        valid_loader = make_loader(valid_windows, params.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.train_net.parameters(), lr=params.learning_rate)
        early_stop = EarlyStopping(patience=params.patience)
        best_state = copy.deepcopy(self.train_net.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

        for epoch in range(params.max_epochs):
            self.train_net.train()
            train_loss = 0.0
            for (batch_windows,) in train_loader:
                batch_windows = batch_windows.to(device)
                optimizer.zero_grad()
                loss = self._forward_loss(batch_windows)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
            info.train_loss_history.append(train_loss / max(len(train_loader), 1))

            val_loss = self._eval_val_loss(valid_loader, device)
            info.val_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = copy.deepcopy(self.train_net.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        self.train_net.load_state_dict(best_state)
        self.pred_net = TimeGradPredictionNetwork(
            input_size=input_size,
            num_layers=1,
            num_cells=max(16, self.num_channels * 8),
            cell_type="LSTM",
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=0.1,
            lags_seq=[1],
            target_dim=self.num_channels,
            conditioning_length=max(16, self.num_channels * 8),
            diff_steps=20,
            loss_type="l2",
            beta_end=0.1,
            beta_schedule="linear",
            residual_layers=4,
            residual_channels=max(8, self.num_channels * 4),
            dilation_cycle_length=2,
            scaling=True,
            num_parallel_samples=1,
        ).to(device)
        self.pred_net.load_state_dict(self.train_net.state_dict(), strict=False)

        self.checkpoint = checkpoints_dir / "timegrad_checkpoint.pt"
        torch.save(self.train_net.state_dict(), self.checkpoint)
        self._is_fitted = True
        return info.as_dict(num_channels=self.num_channels)

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.pred_net is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        device = next(self.pred_net.parameters()).device
        bsz = num_samples
        target_dimension_indicator = (
            torch.arange(self.num_channels, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
            .contiguous()
        )
        past_time_feat = torch.zeros(bsz, self.history_length, self.time_feat_dim, device=device)
        future_time_feat = torch.zeros(bsz, self.prediction_length, self.time_feat_dim, device=device)
        if self.past_target_template is None:
            raise RuntimeError("Missing past_target_template; call fit() first.")
        past_target = self.past_target_template.unsqueeze(0).expand(bsz, -1, -1).to(device).clone()

        # Autoregressive rollout to generation_length (avoids tiling short forecasts).
        chunks: list[torch.Tensor] = []
        remaining = int(generation_length)
        self.pred_net.eval()
        with torch.no_grad():
            while remaining > 0:
                past_observed = torch.ones_like(past_target)
                past_is_pad = torch.zeros(bsz, self.history_length, device=device)
                samples = self.pred_net(
                    target_dimension_indicator=target_dimension_indicator,
                    past_time_feat=past_time_feat,
                    past_target_cdf=past_target,
                    past_observed_values=past_observed,
                    past_is_pad=past_is_pad,
                    future_time_feat=future_time_feat,
                )
                step = samples[:, 0, :, :].float()
                take = min(remaining, step.shape[1])
                chunks.append(step[:, :take, :])
                remaining -= take
                rolled = torch.cat([past_target, step], dim=1)
                past_target = rolled[:, -self.history_length :, :].contiguous()

        data = torch.cat(chunks, dim=1)
        if self.apply_calibration and self.channel_stats is not None:
            data = match_channel_moments(data, self.channel_stats)

        checkpoints = [self.checkpoint] if self.checkpoint is not None else []
        return AdapterGenerateOutput(
            data=data,
            checkpoints=checkpoints,
            logs={"trainer": "timegrad_real"},
            extra_metadata={"num_channels": data.shape[-1]},
        )

