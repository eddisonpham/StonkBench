"""QuantGAN adapter — native multivariate via adapter-layer subclass.

The upstream QuantGANTrainer hardcodes univariate (output_size=1). Rather than
patching the vendor, we subclass QuantGANTrainer and override _init_models(),
fit(), and generate() to support (N, L, C) multivariate windows. The vendor
code is NEVER imported or modified — only the public API is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.utils.device import resolve_device

# Import only the vendor's config/result dataclasses and base trainer
from src.models.deep_learning.quantgan_module import (
    QuantGANConfig,
    QuantGANFitResult,
    QuantGANTrainer,
)


class MultivariateWindowDataset(Dataset):
    """Adapter-layer dataset that preserves (N, L, C) shape for multivariate data."""

    def __init__(self, data: torch.Tensor):
        if data.ndim == 2:
            self.data = data.unsqueeze(-1).float()
        else:
            self.data = data.float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class MultivariateQuantGANTrainer(QuantGANTrainer):
    """Adapter-layer subclass that adds multivariate support to QuantGANTrainer.

    Overrides:
    - _init_models: accepts output_size for C-channel Generator/Discriminator
    - fit: uses MultivariateWindowDataset for (N, L, C) input
    - generate: handles multivariate output without squeezing
    """

    def _init_models(self, output_size: int = 1) -> None:
        self.generator = Generator(self.cfg.noise_dim, output_size).to(self.device)
        self.discriminator = Discriminator(output_size, output_size).to(self.device)

    def fit(
        self,
        train_windows: torch.Tensor,
        valid_windows: Optional[torch.Tensor] = None,
    ) -> QuantGANFitResult:
        train_dataset = MultivariateWindowDataset(train_windows)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.cfg.batch_size, len(train_dataset)),
            shuffle=True,
        )
        valid_loader = None
        if valid_windows is not None and valid_windows.shape[0] > 0:
            valid_dataset = MultivariateWindowDataset(valid_windows)
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.cfg.batch_size, len(valid_dataset)),
                shuffle=False,
            )

        output_size = train_windows.shape[-1] if is_multivariate else 1
        self._init_models(output_size)

        opt_g = optim.RMSprop(self.generator.parameters(), lr=self.cfg.lr)
        opt_d = optim.RMSprop(self.discriminator.parameters(), lr=self.cfg.lr)

        best_val = float("inf")
        best_epoch = 0
        stopped_early = False
        patience_counter = 0
        min_epochs = max(20, self.cfg.patience * 2)
        train_history = []
        val_history = []

        for epoch in range(self.cfg.epochs):
            self.generator.train()
            self.discriminator.train()
            epoch_loss = 0.0
            steps = 0
            for i, real in enumerate(train_loader):
                real = real.to(self.device)
                batch_size, seq_len = real.shape[0], real.shape[1]
                noise = torch.randn(batch_size, seq_len, self.cfg.noise_dim, device=self.device)

                self.discriminator.zero_grad()
                fake = self.generator(noise).detach()
                loss_d = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                loss_d.backward()
                opt_d.step()
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.cfg.clip_value, self.cfg.clip_value)

                if i % self.cfg.d_steps_per_g_step == 0:
                    self.generator.zero_grad()
                    loss_g = -torch.mean(self.discriminator(self.generator(noise)))
                    loss_g.backward()
                    opt_g.step()

                epoch_loss += float((loss_d + loss_g).item())
                steps += 1

            train_history.append(epoch_loss / max(steps, 1))

            if valid_loader is not None:
                val_loss = self._eval_val_loss(valid_loader)
                val_history.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    self._best_generator_state = {
                        k: v.detach().cpu().clone() for k, v in self.generator.state_dict().items()
                    }
                elif epoch + 1 >= min_epochs:
                    patience_counter += 1
                    if patience_counter >= self.cfg.patience:
                        stopped_early = True
                        break
            else:
                best_epoch = epoch + 1
                best_val = train_history[-1]
                self._best_generator_state = {
                    k: v.detach().cpu().clone() for k, v in self.generator.state_dict().items()
                }

        if self._best_generator_state is not None:
            self.generator.load_state_dict(self._best_generator_state)

        return QuantGANFitResult(
            best_val_loss=float(best_val),
            best_epoch=int(best_epoch),
            stopped_early=stopped_early,
            train_loss_history=train_history,
            val_loss_history=val_history,
        )

    def generate(self, num_samples: int, length: int, seed: int = 42) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError("Trainer is not fitted.")
        torch.manual_seed(seed)
        noise = torch.randn(num_samples, length, self.cfg.noise_dim, device=self.device)
        with torch.no_grad():
            fake = self.generator(noise)
        # Only squeeze if univariate (last dim == 1)
        if fake.ndim == 3 and fake.shape[-1] == 1:
            fake = fake.squeeze(-1)
        return fake.float().cpu()


class QuantGANAdapter(ModelAdapter):
    """Multivariate QuantGAN: single joint TCN Generator/Discriminator on full (N, L, C) data.

    Uses adapter-layer MultivariateQuantGANTrainer subclass — vendor code untouched.
    """

    model_name = "QuantGAN"

    def __init__(self) -> None:
        super().__init__()
        self.trainer: MultivariateQuantGANTrainer | None = None
        self.base_length: int = 1
        self.num_channels: int = 1
        self.checkpoints: list[Path] = []

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("QuantGANAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("QuantGANAdapter requires non-empty valid_windows.")

        from src.experiments.adapters.deep_learning.training_utils import parse_training_params

        params = parse_training_params(fit_input)
        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])

        self.trainer = MultivariateQuantGANTrainer(
            device=fit_input.device,
            cfg=QuantGANConfig(
                epochs=params.max_epochs,
                batch_size=max(8, min(params.batch_size, windows.shape[0])),
                lr=params.learning_rate,
                patience=params.patience,
            ),
        )
        fit_result = self.trainer.fit(windows, valid_windows)

        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": self.num_channels,
                "base_length": self.base_length,
                "state_dict": self.trainer.generator.state_dict(),
                "discriminator_state_dict": self.trainer.discriminator.state_dict(),
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]
        self._is_fitted = True

        return {
            "num_channels": self.num_channels,
            "best_val_loss": fit_result.best_val_loss,
            "best_epoch": fit_result.best_epoch,
            "stopped_early": fit_result.stopped_early,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.trainer is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.trainer.generate(num_samples, self.base_length, seed=seed)
        # Ensure output is always 3-D (R, L, C) for downstream pipeline consistency
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            checkpoints=self.checkpoints,
            logs={"trainer": "quantgan"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
