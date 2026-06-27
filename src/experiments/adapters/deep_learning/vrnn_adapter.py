from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.calibration import (
    ChannelMomentStats,
    match_channel_moments,
    shuffle_time_within_windows,
)
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
    use_calibration,
)
from src.models.deep_learning.VariationalRecurrentNeuralNetwork.model import VRNN
from src.utils.artifact_utils import stitch_sequences


class VRNNAdapter(ModelAdapter):
    """Multivariate VRNN adapter using model-folder implementation."""

    model_name = "VRNN"

    def __init__(self) -> None:
        super().__init__()
        self.model: VRNN | None = None
        self.device = "cpu"
        self.num_channels = 1
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.channel_stats: ChannelMomentStats | None = None
        self.scale_min: torch.Tensor | None = None
        self.scale_max: torch.Tensor | None = None
        self.apply_calibration = False

    def _scale_batch(self, batch_x: torch.Tensor) -> torch.Tensor:
        denom = (self.scale_max - self.scale_min).clamp(min=1e-8)
        return ((batch_x - self.scale_min) / denom).clamp(0.0, 1.0)

    def _forward_loss(self, batch_x: torch.Tensor) -> torch.Tensor:
        batch_x = batch_x.transpose(0, 1)
        batch_x = self._scale_batch(batch_x)
        kld_loss, nll_loss, _, _ = self.model(batch_x)
        return kld_loss + nll_loss

    @torch.no_grad()
    def _eval_val_loss(self, loader: DataLoader, device: torch.device) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            total += float(self._forward_loss(batch_x).item())
            count += 1
        return total / max(count, 1)

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("VRNNAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("VRNNAdapter requires non-empty valid_windows for model selection.")

        params = parse_training_params(fit_input)
        n_samples, seq_len, channels = windows.shape
        self.base_length = seq_len
        self.num_channels = channels
        device = resolve_device(fit_input.device)
        self.device = str(device)
        self.channel_stats = ChannelMomentStats.from_windows(windows)
        self.apply_calibration = use_calibration(fit_input)
        self.scale_min = windows.amin(dim=(0, 1))
        self.scale_max = windows.amax(dim=(0, 1))

        self.model = VRNN(
            x_dim=channels,
            h_dim=max(32, channels * 16),
            z_dim=max(8, channels * 4),
            n_layers=1,
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)

        train_loader = make_loader(windows, params.batch_size, shuffle=True)
        valid_loader = make_loader(valid_windows, params.batch_size, shuffle=False)
        early_stop = EarlyStopping(patience=params.patience)
        best_state = copy.deepcopy(self.model.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

        for epoch in range(params.max_epochs):
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                loss = self._forward_loss(batch_x)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                train_loss += float(loss.item())
            info.train_loss_history.append(train_loss / max(len(train_loader), 1))

            val_loss = self._eval_val_loss(valid_loader, device)
            info.val_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        self.model.load_state_dict(best_state)
        ckpt_path = checkpoints_dir / "vrnn_checkpoint.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        self.checkpoints = [ckpt_path]
        self._is_fitted = True
        return info.as_dict(num_channels=self.num_channels, base_length=self.base_length)

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        generated = []
        denom = (self.scale_max - self.scale_min).clamp(min=1e-8)
        with torch.no_grad():
            for _ in range(num_samples):
                seq = self.model.sample(self.base_length).unsqueeze(0)
                seq = seq * denom.view(1, 1, -1) + self.scale_min.view(1, 1, -1)
                generated.append(seq)

        data = torch.cat(generated, dim=0).float()
        if self.apply_calibration and self.channel_stats is not None:
            data = match_channel_moments(data, self.channel_stats)
            data = shuffle_time_within_windows(data, seed)
        if generation_length != data.shape[1]:
            stitched = []
            for c in range(data.shape[-1]):
                stitched_c = stitch_sequences(data[:, :, c], generation_length, seed + c)
                stitched.append(stitched_c.unsqueeze(-1))
            data = torch.cat(stitched, dim=-1)

        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"generator": "vrnn"},
            extra_metadata={"num_channels": self.num_channels},
        )
