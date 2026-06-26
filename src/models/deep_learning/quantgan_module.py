"""
QuantGAN core module extracted from QuantGAN.ipynb.

This keeps the original TCN generator/discriminator architecture pattern while
providing a clean training/generation API for adapter integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_hidden,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding="same",
        )
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(
            n_hidden,
            n_outputs,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding="same",
        )
        self.relu2 = nn.PReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self._init_weights()

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            if i > 1:
                dilation = 2 * dilation
            layers.append(TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation))
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2))
        return self.conv(y).transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, noise_dim: int, output_size: int):
        super().__init__()
        self.net = TCN(noise_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class Discriminator(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class UnivariateWindowDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        if data.ndim != 2:
            raise ValueError("QuantGAN dataset expects shape (N, L) windows")
        self.data = data.unsqueeze(-1).float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


@dataclass
class QuantGANConfig:
    noise_dim: int = 3
    lr: float = 2e-4
    clip_value: float = 0.01
    batch_size: int = 30
    epochs: int = 3
    d_steps_per_g_step: int = 5


class QuantGANTrainer:
    def __init__(self, device: str = "cpu", cfg: Optional[QuantGANConfig] = None):
        self.device = device
        self.cfg = cfg or QuantGANConfig()
        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None

    def fit(self, train_windows: torch.Tensor) -> None:
        dataset = UnivariateWindowDataset(train_windows)
        loader = DataLoader(dataset, batch_size=min(self.cfg.batch_size, len(dataset)), shuffle=True)

        self.generator = Generator(self.cfg.noise_dim, 1).to(self.device)
        self.discriminator = Discriminator(1, 1).to(self.device)
        opt_g = optim.RMSprop(self.generator.parameters(), lr=self.cfg.lr)
        opt_d = optim.RMSprop(self.discriminator.parameters(), lr=self.cfg.lr)

        for _ in range(self.cfg.epochs):
            for i, real in enumerate(loader):
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

    def generate(self, num_samples: int, length: int, seed: int = 42) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError("Trainer is not fitted.")
        torch.manual_seed(seed)
        noise = torch.randn(num_samples, length, self.cfg.noise_dim, device=self.device)
        with torch.no_grad():
            fake = self.generator(noise).squeeze(-1)
        return fake.float().cpu()

