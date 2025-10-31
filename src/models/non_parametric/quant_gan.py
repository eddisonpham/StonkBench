"""
PyTorch implementation of QuantGAN.

This module defines a QuantGAN model that inherits from DeepLearningModel
and follows the formatting/style used by parametric models. It handles
log returns without normalization and suppresses print outputs.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base.base_model import DeepLearningModel


class _TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu2 = nn.PReLU()

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class _TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            if i > 1:
                dilation *= 2
            layers.append(_TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation))
        
        self.net = nn.Sequential(*layers)
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self._init_weights()

    def _init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2))
        return self.conv(y).transpose(1, 2)


class _Generator(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        self.net = _TCN(input_size, output_size, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class _Discriminator(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        self.net = _TCN(input_size, output_size, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class QuantGAN(DeepLearningModel):
    """
    QuantGAN implemented with PyTorch.

    - Inherits from DeepLearningModel and follows its interface.
    - Expects fixed-length sequences from the provided DataLoader in fit.
    - Handles log returns without normalization and does not print training logs.
    """

    def __init__(
        self,
        length: int,
        num_channels: int,
        embedding_dim: int = 3,
        hidden_dim: int = 80,
        lr: float = 0.0002,
        clip_value: float = 0.01,
        n_critic: int = 5,
    ):
        super().__init__(length=length, num_channels=num_channels)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.clip_value = float(clip_value)
        self.n_critic = int(n_critic)

        # Networks
        self.generator = _Generator(embedding_dim, num_channels, hidden_dim)
        self.discriminator = _Discriminator(num_channels, 1, hidden_dim)

        # Optimizers
        self.opt_G = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.opt_D = optim.RMSprop(self.discriminator.parameters(), lr=lr)

        self.to(self.device)

    def _sample_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.embedding_dim, device=self.device)

    def fit(
        self,
        data_loader,
        num_epochs: int = 10,
    ):
        self.train()

        for epoch in range(num_epochs):
            for i, real_batch in enumerate(data_loader):
                if isinstance(real_batch, (list, tuple)):
                    real_batch = real_batch[0]
                real = real_batch.to(self.device)  # (B, L, N)
                batch_size, seq_len, _ = real.shape

                # Sample noise
                z = self._sample_noise(batch_size, seq_len)

                # Train Discriminator
                self.opt_D.zero_grad(set_to_none=True)
                fake = self.generator(z).detach()
                
                loss_D = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                loss_D.backward()
                self.opt_D.step()

                # Weight clipping for Lipschitz constraint
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # Train Generator (every n_critic steps)
                if i % self.n_critic == 0:
                    self.opt_G.zero_grad(set_to_none=True)
                    fake = self.generator(z)
                    loss_G = -torch.mean(self.discriminator(fake))
                    loss_G.backward()
                    self.opt_G.step()

            print(f"QuantGAN epoch {epoch + 1}/{num_epochs}")

        self.eval()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        seq_length: Optional[int] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        if seq_length is None:
            seq_length = self.length
        torch.manual_seed(seed)
        z = self._sample_noise(num_samples, seq_length)
        fake = self.generator(z)
        return fake.detach().cpu()