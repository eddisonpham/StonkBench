"""TimeGrad adapter — native multivariate (single joint network on full (N, L, C) data).

Vendor: kongqi404/timegrad vendored at src/models/deep_learning/timegrad/.
We import the upstream TimeGradTrainingNetwork + GaussianDiffusion + EpsilonTheta + utils directly.

Strategy: single joint TimeGradTrainingNetwork with target_dim=C, processing the full
multivariate tensor. This preserves cross-channel correlations during training and generation.

Internal window sizes (smaller than StonkBench's preprocessed L=252; we
take the LAST timegrad_total timesteps of each window):
  context_length = 24
  prediction_length = 24
  lags_seq = [1, 24, 168]
  history_length = context_length + max(lags_seq) = 192
  timegrad_total = history_length + prediction_length = 216
The adapter slices each StonkBench window's last 216 timesteps before
feeding to TimeGrad's forward().
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
)


def _import_timegrad():
    root = Path(__file__).resolve().parents[3] / "models" / "deep_learning" / "timegrad"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from module import GaussianDiffusion, DiffusionOutput  # noqa: F401
    from epsilon_theta import EpsilonTheta  # noqa: F401
    from time_grad_network import TimeGradTrainingNetwork, TimeGradPredictionNetwork
    from utils import weighted_average  # noqa: F401
    return TimeGradTrainingNetwork, TimeGradPredictionNetwork


class TimeGradAdapter(ModelAdapter):
    """Multivariate TimeGrad: single joint network processing full (N, L, C) data.

    The upstream TimeGradTrainingNetwork natively supports target_dim > 1 via
    its embedding layer and diffusion model. We use target_dim=C to process
    all channels jointly, preserving cross-channel correlations.
    """

    model_name = "TimeGrad"
    supports_arbitrary_generation = True

    # Vendor default config (matches kongqi404/timegrad TimeGradEstimator defaults).
    _NUM_CELLS = 40
    _NUM_LAYERS = 2
    _CELL_TYPE = "LSTM"
    _DIFF_STEPS = 100
    _LOSS_TYPE = "l2"
    _BETA_END = 0.1
    _BETA_SCHEDULE = "linear"
    _RESIDUAL_LAYERS = 8
    _RESIDUAL_CHANNELS = 8
    _DILATION_CYCLE = 2
    # STONKBENCH_PATCH_2026-07-28_G: derive from _NUM_CELLS so vendor-family
    # changes do not silently desync.

    _NUM_PARALLEL_SAMPLES = 100
    _DROPOUT_RATE = 0.1

    def __init__(self) -> None:
        super().__init__()
        # STONKBENCH_PATCH_2026-07-28_J1: dynamic init-time coupling replaces the
        # static class-body _CONDITIONING_LENGTH = _NUM_CELLS so runtime mutation
        # of _NUM_CELLS propagates. Assert keeps the invariant explicit.
        self._CONDITIONING_LENGTH = self._NUM_CELLS
        assert self._CONDITIONING_LENGTH == self._NUM_CELLS, (
            f"_CONDITIONING_LENGTH ({self._CONDITIONING_LENGTH}) != _NUM_CELLS ({self._NUM_CELLS}). "
            "Re-bind in __init__ before constructing TimeGradTrainingNetwork."
        )
        self.TimeGradTrainingNetwork, self.TimeGradPredictionNetwork = _import_timegrad()
        self.model: Any = None  # single joint TimeGradTrainingNetwork
        self.device = "cpu"
        self.num_channels = 1
        self.base_length = 252
        self.context_length = 24
        self.prediction_length = 24
        self.lags_seq = [1, 24, 168]
        self.checkpoints: List[Path] = []

    @property
    def history_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    @property
    def timegrad_total(self) -> int:
        return self.history_length + self.prediction_length

    def _make_gluonts_args(self, windows: torch.Tensor, device: torch.device | None = None):
        """Build gluonts-style args for multivariate (N, L, C) windows."""
        N, L, C = windows.shape
        history = self.history_length
        dev = device if device is not None else windows.device

        past_target_cdf = windows[:, :history, :]  # (N, history, C)
        future_target_cdf = windows[:, history:history + self.prediction_length, :]  # (N, pred, C)
        past_observed_values = torch.ones_like(past_target_cdf)
        future_observed_values = torch.ones_like(future_target_cdf)
        past_is_pad = torch.zeros(N, history, dtype=torch.float32, device=dev)
        past_time_feat = torch.zeros(N, history, 1, dtype=torch.float32, device=dev)
        future_time_feat = torch.zeros(N, self.prediction_length, 1, dtype=torch.float32, device=dev)
        # target_dimension_indicator: (N, C) — one index per channel
        target_dimension_indicator = torch.arange(C, device=dev).unsqueeze(0).expand(N, -1)
        return (
            target_dimension_indicator, past_time_feat, past_target_cdf,
            past_observed_values, past_is_pad, future_time_feat,
            future_target_cdf, future_observed_values,
        )

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("TimeGradAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("TimeGradAdapter requires non-empty valid_windows.")

        params = parse_training_params(fit_input)
        n_windows, seq_len, channels = windows.shape
        # TimeGrad requires at least timegrad_total=216 timesteps per window
        if seq_len < self.timegrad_total:
            raise ValueError(
                f"TimeGradAdapter requires seq_len >= timegrad_total={self.timegrad_total} "
                f"(history {self.history_length} + prediction {self.prediction_length}); "
                f"got seq_len={seq_len}. Increase --generation_length to >= {self.timegrad_total}."
            )
        self.num_channels = channels
        self.base_length = seq_len

        device = resolve_device(fit_input.device)
        self.device = str(device)

        per_window_train = windows[:, -self.timegrad_total:, :].contiguous()
        per_window_valid = valid_windows[:, -self.timegrad_total:, :].contiguous() if valid_windows.numel() > 0 else per_window_train

        # input_size = lags_seq * target_dim + target_dim * embed_dim + num_time_feat
        # For target_dim=C, embed_dim=1, num_time_feat=1, lags_seq len=3 -> 3*C + C*1 + 1
        input_size = len(self.lags_seq) * channels + channels * 1 + 1

        net = self.TimeGradTrainingNetwork(
            input_size=input_size,
            num_layers=self._NUM_LAYERS,
            num_cells=self._NUM_CELLS,
            cell_type=self._CELL_TYPE,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self._DROPOUT_RATE,
            lags_seq=list(self.lags_seq),
            target_dim=channels,  # <-- joint multivariate
            conditioning_length=self._CONDITIONING_LENGTH,
            diff_steps=self._DIFF_STEPS,
            loss_type=self._LOSS_TYPE,
            beta_end=self._BETA_END,
            beta_schedule=self._BETA_SCHEDULE,
            residual_layers=self._RESIDUAL_LAYERS,
            residual_channels=self._RESIDUAL_CHANNELS,
            dilation_cycle_length=self._DILATION_CYCLE,
            cardinality=[1] * channels,  # <-- per-channel cardinality
            embedding_dimension=1,  # <-- per-channel index embeddings (vendor default)
            scaling=False,  # StonkBench data already z-scored
        ).to(device)

        # Vendor-faithful: plain Adam (no weight_decay override; vendor does not tune it).
        # NO OneCycleLR — vendor's PyTorch Lightning Trainer manages its own LR schedule.
        optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
        num_batches_per_epoch = 50
        train_loader = make_loader(per_window_train, params.batch_size, shuffle=True)
        valid_loader = make_loader(per_window_valid, params.batch_size, shuffle=False) if per_window_valid.shape[0] > 0 else train_loader

        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_val_loss = float("inf")
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

        for epoch in range(params.max_epochs):
            net.train()
            train_loss = 0.0
            # Vendor uses num_batches_per_epoch=50 per epoch
            train_iter = iter(train_loader)
            for _ in range(num_batches_per_epoch):
                try:
                    (batch_x,) = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    (batch_x,) = next(train_iter)
                batch_x = batch_x.to(device)
                args = self._make_gluonts_args(batch_x, device=device)
                optimizer.zero_grad()
                out = net(*args)
                loss = out[0] if isinstance(out, (tuple, list)) else out
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
                train_loss += float(loss.item())
            avg_train = train_loss / num_batches_per_epoch

            net.eval()
            val_loss = 0.0
            nv = 0
            with torch.no_grad():
                for (batch_x,) in valid_loader:
                    batch_x = batch_x.to(device)
                    args = self._make_gluonts_args(batch_x, device=device)
                    out = net(*args)
                    loss = out[0] if isinstance(out, (tuple, list)) else out
                    val_loss += float(loss.item())
                    nv += 1
            avg_val = val_loss / max(nv, 1)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            info.train_loss_history.append(avg_train)
            info.val_loss_history.append(avg_val)

        net.load_state_dict(best_state)
        self.model = net

        info.best_val_loss = best_val_loss
        info.best_epoch = params.max_epochs

        # One consolidated FINAL checkpoint labeled with seq length
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": self.num_channels,
                "base_length": self.base_length,
                "history_length": self.history_length,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "lags_seq": list(self.lags_seq),
                "num_cells": self._NUM_CELLS,
                "num_layers": self._NUM_LAYERS,
                "cell_type": self._CELL_TYPE,
                "diff_steps": self._DIFF_STEPS,
                "loss_type": self._LOSS_TYPE,
                "beta_end": self._BETA_END,
                "beta_schedule": self._BETA_SCHEDULE,
                "residual_layers": self._RESIDUAL_LAYERS,
                "residual_channels": self._RESIDUAL_CHANNELS,
                "dilation_cycle_length": self._DILATION_CYCLE,
                "conditioning_length": self._CONDITIONING_LENGTH,
                "num_parallel_samples": self._NUM_PARALLEL_SAMPLES,
                "target_dim": self.num_channels,
                "cardinality": [1] * self.num_channels,
                "embedding_dimension": self.num_channels,
                "state_dict": net.state_dict(),
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]
        self._is_fitted = True
        return info.as_dict(
            num_channels=self.num_channels,
            base_length=self.base_length,
            history_length=self.history_length,
            prediction_length=self.prediction_length,
            num_cells=self._NUM_CELLS,
            num_layers=self._NUM_LAYERS,
        )

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        C = self.num_channels
        input_size = len(self.lags_seq) * C + C * 1 + 1

        device_for_c = next(self.model.parameters()).device
        pred = self.TimeGradPredictionNetwork(
            num_parallel_samples=self._NUM_PARALLEL_SAMPLES,
            input_size=input_size,
            num_layers=self._NUM_LAYERS,
            num_cells=self._NUM_CELLS,
            cell_type=self._CELL_TYPE,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=0.0,
            lags_seq=list(self.lags_seq),
            target_dim=C,
            conditioning_length=self._CONDITIONING_LENGTH,
            diff_steps=self._DIFF_STEPS,
            loss_type=self._LOSS_TYPE,
            beta_end=self._BETA_END,
            beta_schedule=self._BETA_SCHEDULE,
            residual_layers=self._RESIDUAL_LAYERS,
            residual_channels=self._RESIDUAL_CHANNELS,
            dilation_cycle_length=self._DILATION_CYCLE,
            cardinality=[1] * C,
            embedding_dimension=1,  # <-- per-channel index embeddings (vendor default)
            scaling=False,
        )
        pred.load_state_dict(self.model.state_dict())
        pred.eval().to(device_for_c)

        # --- Autoregressive rollout ---
        # TimeGrad predicts prediction_length=24 steps per forward pass.
        # To generate arbitrary lengths, we roll the past context window
        # forward by prediction_length each step and concatenate outputs.
        target_dimension_indicator = torch.arange(C, device=device_for_c).unsqueeze(0).expand(num_samples, -1)
        past_target_cdf = torch.zeros(num_samples, self.history_length, C, device=device_for_c)
        past_observed_values = torch.ones_like(past_target_cdf)
        past_is_pad = torch.zeros(num_samples, self.history_length, device=device_for_c)
        past_time_feat = torch.zeros(num_samples, self.history_length, 1, device=device_for_c)

        # Always generate in chunks of prediction_length (model's fixed
        # internal dimension) and trim to generation_length at the end.
        # This avoids passing a smaller future_time_feat than the model
        # was constructed with, which would cause a shape mismatch.
        future_time_feat = torch.zeros(num_samples, self.prediction_length, 1, device=device_for_c)
        generated_chunks: list[torch.Tensor] = []
        remaining = generation_length
        with torch.no_grad():
            while remaining > 0:
                sample_paths = pred(
                    target_dimension_indicator=target_dimension_indicator,
                    past_time_feat=past_time_feat,
                    past_target_cdf=past_target_cdf,
                    past_observed_values=past_observed_values,
                    past_is_pad=past_is_pad,
                    future_time_feat=future_time_feat,
                )
                # sample_paths: (batch, num_parallel, prediction_length, C)
                chunk = sample_paths[:, 0, :, :]  # (batch, prediction_length, C)
                take = min(self.prediction_length, remaining)
                generated_chunks.append(chunk[:, :take, :])
                remaining -= take
                # Roll past context: shift left by prediction_length, append full chunk
                past_target_cdf = torch.cat([past_target_cdf[:, self.prediction_length:, :], chunk], dim=1)
                past_observed_values = torch.ones_like(past_target_cdf)
                past_is_pad = torch.zeros(num_samples, past_target_cdf.shape[1], device=device_for_c)
                past_time_feat = torch.zeros(num_samples, past_target_cdf.shape[1], 1, device=device_for_c)

        out = torch.cat(generated_chunks, dim=1).float().cpu()  # (batch, generation_length, C)

        # Vendor-faithful: no post-hoc moment injection. Output is whatever the model produces.
        return AdapterGenerateOutput(
            data=out,
            checkpoints=self.checkpoints,
            logs={"generator": "timegrad"},
            extra_metadata={
                "num_channels": self.num_channels,
                "history_length": self.history_length,
                "prediction_length": self.prediction_length,
                "num_cells": self._NUM_CELLS,
                "num_layers": self._NUM_LAYERS,
                "lags_seq": list(self.lags_seq),
                "diff_steps": self._DIFF_STEPS,
            },
        )
