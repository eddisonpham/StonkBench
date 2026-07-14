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


class VRNNAdapter(ModelAdapter):
    """Multivariate VRNN on continuous z-scored returns (Gaussian emission)."""

    model_name = "VRNN"

    def __init__(self) -> None:
        super().__init__()
        self.model: VRNN | None = None
        self.device = "cpu"
        self.num_channels = 1
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.channel_stats: ChannelMomentStats | None = None
        self.apply_calibration = False

    def _forward_loss(self, batch_x: torch.Tensor, kl_weight: float) -> torch.Tensor:
        # VRNN expects time-major (T, B, C)
        batch_x = batch_x.transpose(0, 1)
        kld_loss, nll_loss, _, _ = self.model(batch_x)
        return nll_loss + float(kl_weight) * kld_loss

    @torch.no_grad()
    def _eval_val_loss(self, loader: DataLoader, device: torch.device, kl_weight: float) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            total += float(self._forward_loss(batch_x, kl_weight).item())
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

        # Modest capacity: channels*16 was ~800 and overfit/collapsed with tiny batches.
        h_dim = max(64, min(256, channels * 4))
        z_dim = max(16, min(64, channels * 2))
        self.model = VRNN(
            x_dim=channels,
            h_dim=h_dim,
            z_dim=z_dim,
            n_layers=1,
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)

        train_loader = make_loader(windows, params.batch_size, shuffle=True)
        valid_loader = make_loader(valid_windows, params.batch_size, shuffle=False)
        early_stop = EarlyStopping(patience=params.patience, min_epochs=max(20, params.patience * 2))
        best_state = copy.deepcopy(self.model.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)
        kl_warmup = max(10, params.max_epochs // 5)

        for epoch in range(params.max_epochs):
            kl_weight = min(1.0, float(epoch + 1) / float(kl_warmup))
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                loss = self._forward_loss(batch_x, kl_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                train_loss += float(loss.item())
            info.train_loss_history.append(train_loss / max(len(train_loader), 1))

            val_loss = self._eval_val_loss(valid_loader, device, kl_weight)
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
        return info.as_dict(
            num_channels=self.num_channels,
            base_length=self.base_length,
            h_dim=h_dim,
            z_dim=z_dim,
        )

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        length = int(generation_length)
        generated = []
        with torch.no_grad():
            for _ in range(num_samples):
                seq = self.model.sample(length).unsqueeze(0)
                generated.append(seq)

        data = torch.cat(generated, dim=0).float().cpu()
        if self.apply_calibration and self.channel_stats is not None:
            data = match_channel_moments(data, self.channel_stats)

        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"generator": "vrnn"},
            extra_metadata={"num_channels": self.num_channels},
        )
