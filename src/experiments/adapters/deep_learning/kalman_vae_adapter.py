from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _import_kvae():
    """Import the vendored Kalman-VAE package on first use."""
    root = Path(__file__).resolve().parents[3] / "models" / "deep_learning" / "kalman-vae"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from kvae.kalman_vae import KalmanVariationalAutoencoder
    from kvae.sample_control import SampleControl

    return KalmanVariationalAutoencoder, SampleControl


class KalmanVAEAdapter(ModelAdapter):
    """Kalman-VAE adapter for multivariate z-scored log-returns.

    The original K-VAE repository targets image/video data.  To keep the
    integration vendor-faithful, we map each 1-D observation (a vector of
    ``C`` channels) onto a tiny square grayscale image whose side is the
    smallest multiple of 8 that can hold ``C`` values.  Three stride-2
    Conv2d layers reduce such a square to a 1x1 feature map, and the
    matching transposed-conv decoder expands back to the same square, so
    the architecture remains completely unmodified.
    """

    model_name = "KalmanVAE"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.device = "cpu"
        self.num_channels = 1
        self.base_length = 1
        self.a_dim = 16
        self.z_dim = 8
        self.image_side = 8
        self.checkpoints: List[Path] = []

        self.KalmanVariationalAutoencoder, self.SampleControl = _import_kvae()

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _to_model_input(self, windows: torch.Tensor) -> torch.Tensor:
        """(N, L, C) -> (L, N, 1, H, W) with H=W=image_side and C <= H*W."""
        N, L, C = windows.shape
        if self.image_side * self.image_side < C:
            raise ValueError(
                f"Cannot fit {C} channels into {self.image_side}x{self.image_side} image"
            )
        pad = self.image_side * self.image_side - C
        if pad:
            padded = torch.nn.functional.pad(
                windows.reshape(N, L, C), (0, pad), value=0.0
            )
        else:
            padded = windows
        # (N, L, H, W)
        img = padded.view(N, L, self.image_side, self.image_side)
        # (L, N, H, W) -> (L, N, 1, H, W)
        return img.permute(1, 0, 2, 3).unsqueeze(2)

    def _from_model_output(self, x: torch.Tensor, num_samples: int, channels: int) -> torch.Tensor:
        """(L*N, 1, H, W) -> (N, L, C)."""
        if x.ndim != 4:
            raise ValueError(f"Expected decoder output of shape (L*N, 1, H, W), got {x.shape}")
        L_total, one, H, W = x.shape
        if one != 1 or H != self.image_side or W != self.image_side:
            raise ValueError(f"Unexpected decoder output shape {x.shape}")
        # (L, N, H*W) -> (L, N, C)
        flat = x.view(L_total // num_samples, num_samples, H * W)
        if flat.shape[-1] < channels:
            raise ValueError(
                f"Decoder output has fewer pixels ({flat.shape[-1]}) than channels ({channels})"
            )
        return flat[..., :channels].permute(1, 0, 2)

    def _sample_control(self, sample: bool = True) -> Any:
        sc = self.SampleControl
        return sc(
            encoder="sample" if sample else "mean",
            decoder="mean",
            state_transition="sample" if sample else "mean",
            observation="sample" if sample else "mean",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _forward_loss(self, batch_x: torch.Tensor) -> torch.Tensor:
        xs = self._to_model_input(batch_x)  # (L, N, 1, H, W)
        objective, _ = self.model.elbo(
            self._sample_control(sample=True),
            xs=xs,
            reconst_weight=1.0,
            regularization_weight=1.0,
            kalman_weight=1.0,
            kl_weight=0.0,
            learn_weight_model=True,
            symmetrize_covariance=True,
            burn_in=0,
            sequence_operation="mean",
            batch_operation="mean",
        )
        return -objective

    @torch.no_grad()
    def _eval_val_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        device = next(self.model.parameters()).device
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            total += float(self._forward_loss(batch_x.to(device)).item())
            count += 1
        return total / max(count, 1)

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("KalmanVAEAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("KalmanVAEAdapter requires non-empty valid_windows for model selection.")

        params = parse_training_params(fit_input)
        n_samples, seq_len, channels = windows.shape
        self.base_length = seq_len
        self.num_channels = channels

        # Choose the smallest square whose side is a multiple of 8 and can
        # hold all channels.  The K-VAE encoder/decoder uses 3 stride-2 layers,
        # so an 8x8 input collapses to 1x1 and expands back to 8x8.
        side = max(8, int(math.ceil(math.sqrt(channels) / 8) * 8))
        self.image_side = side

        device = resolve_device(fit_input.device)
        self.device = str(device)

        # Allow hyperparameter overrides from metadata, otherwise use defaults
        # tuned for small 1x1-image financial series.
        meta = fit_input.metadata
        self.a_dim = int(meta.get("kvae_a_dim", 16))
        self.z_dim = int(meta.get("kvae_z_dim", 8))
        k = int(meta.get("kvae_K", 3))
        dynamics_net = str(meta.get("kvae_dynamics", "lstm"))

        self.model = self.KalmanVariationalAutoencoder(
            image_size=(self.image_side, self.image_side),
            image_channels=1,
            a_dim=self.a_dim,
            z_dim=self.z_dim,
            K=k,
            init_transition_reg_weight=0.9,
            init_observation_reg_weight=0.9,
            learn_noise_covariance=True,
            init_noise_scale=1.0,
            dynamics_parameter_network=dynamics_net,
            decoder_type="gaussian",
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)
        train_loader = make_loader(windows, params.batch_size, shuffle=True)
        valid_loader = make_loader(valid_windows, params.batch_size, shuffle=False)
        early_stop = EarlyStopping(patience=params.patience, min_epochs=max(20, params.patience * 2))
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                train_loss += float(loss.item())
            info.train_loss_history.append(train_loss / max(len(train_loader), 1))

            val_loss = self._eval_val_loss(valid_loader)
            info.val_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        self.model.load_state_dict(best_state)
        ckpt_path = checkpoints_dir / "kalman_vae_checkpoint.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        self.checkpoints = [ckpt_path]
        self._is_fitted = True
        # Persist ONE consolidated FINAL checkpoint labeled with the seq
        # length so downstream regeneration has a single canonical ckpt per
        # (model, seq_length). The pre-existing kalman_vae_checkpoint.pt
        # stays for backward compat with consumers that still look for it.
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(self.model.state_dict(), final_ckpt)

        return info.as_dict(
            num_channels=self.num_channels,
            base_length=self.base_length,
            a_dim=self.a_dim,
            z_dim=self.z_dim,
            image_side=self.image_side,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        device = next(self.model.parameters()).device

        # Seed with a single latent observation a_0 ~ N(0, I).
        a0 = torch.randn(1, num_samples, self.a_dim, device=device)

        with torch.no_grad():
            # Run the Kalman filter on the single-step prefix to obtain the
            # initial state estimates required by predict_future.
            weight_model = self.model.state_space_model.weight_model
            if hasattr(weight_model, "clear_hidden_state"):
                weight_model.clear_hidden_state()

            (
                filter_means,
                filter_covariances,
                filter_next_means,
                filter_next_covariances,
                mat_As,
                mat_Cs,
                as_resampled,
                weights,
            ) = self.model.state_space_model.kalman_filter(
                a0,
                sample_control=self._sample_control(sample=True),
                observation_mask=None,
                learn_weight_model=False,
                symmetrize_covariance=True,
                burn_in=0,
            )

            # Reset the dynamics LSTM hidden state so predict_future starts from
            # a clean prefix and not from the filter's internal state.
            if hasattr(weight_model, "clear_hidden_state"):
                weight_model.clear_hidden_state()

            # Predict the remaining L-1 future latent observations (the prefix
            # itself counts as step 0, so we ask for generation_length steps).
            future_out = self.model.state_space_model.predict_future(
                a0,
                filter_means,
                filter_covariances,
                filter_next_means,
                filter_next_covariances,
                mat_As,
                mat_Cs,
                num_steps=generation_length,
                sample_control=self._sample_control(sample=True),
            )
            as_tensor = future_out[0]  # (1 + generation_length, N, a_dim)

        # Drop the seeding step and keep the generated future.
        gen_a = as_tensor[-generation_length:]  # (L, N, a_dim)
        with torch.no_grad():
            decoded = self.model.decoder(gen_a.view(-1, self.a_dim))
            gen_x = decoded.sample()  # (L*N, 1, H, W)

        data = self._from_model_output(
            gen_x, num_samples=num_samples, channels=self.num_channels
        )
        # Vendor-faithful: no post-hoc moment injection. Output is whatever the model produces.
        return AdapterGenerateOutput(
            data=data.float().cpu(),
            checkpoints=self.checkpoints,
            logs={"generator": "kalman_vae"},
            extra_metadata={"num_channels": self.num_channels, "a_dim": self.a_dim, "z_dim": self.z_dim},
        )
