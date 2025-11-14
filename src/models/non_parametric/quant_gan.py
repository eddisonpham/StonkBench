"""
PyTorch implementation of QuantGAN.

This module defines a QuantGAN model that inherits from DeepLearningModel
and follows the formatting/style used by parametric models. It handles
log returns without normalization and suppresses print outputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base.base_model import DeepLearningModel


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
    ):
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

    def _init_weights(self):
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)

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
                dilation *= 2
            else:
                dilation = 1
            layers.append(TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation))
        
        self.net = nn.Sequential(*layers)
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2))
        return self.conv(y).transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        self.net = TCN(input_size, output_size, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class Discriminator(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden: int = 80):
        super().__init__()
        self.net = TCN(input_size, output_size, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantGAN(DeepLearningModel):
    """
    QuantGAN implemented with PyTorch.

    - Inherits from DeepLearningModel and follows its interface.
    - Expects batches shaped `(batch_size, seq_len)` or `(batch_size, seq_len, 1)`.
    - Uses a standard GAN objective with BCE loss.
    """

    def __init__(
        self,
        num_channels: int = 1,
        embedding_dim: int = 3,
        hidden_dim: int = 80,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.9,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.num_channels = int(num_channels)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(self.embedding_dim, self.num_channels, self.hidden_dim).to(self.device)
        self.discriminator = Discriminator(self.num_channels, 1, self.hidden_dim).to(self.device)

        self.opt_G = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.opt_D = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.adv_loss = nn.BCEWithLogitsLoss()

    def _ensure_sequence_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            if x.size(-1) != self.num_channels:
                raise ValueError(
                    f"Expected last dimension to equal num_channels={self.num_channels}, got {x.size(-1)}."
                )
        else:
            raise ValueError(
                "QuantGAN expects batches shaped (batch_size, seq_len) or (batch_size, seq_len, channels)."
            )
        return x

    def _sample_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.embedding_dim, device=self.device)

    def fit(
        self,
        data_loader,
        num_epochs: int = 10,
        *args,
        **kwargs
    ):
        self.train()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(num_epochs):
            for i, real_batch in enumerate(data_loader):
                real = real_batch.float().to(self.device)
                real = self._ensure_sequence_shape(real)

                batch_size, seq_len, _ = real.shape

                valid = torch.ones(batch_size, seq_len, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, seq_len, 1, device=self.device)

                # === Train Discriminator ===
                self.opt_D.zero_grad(set_to_none=True)
                z = self._sample_noise(batch_size, seq_len)
                fake = self.generator(z).detach()

                real_logits = self.discriminator(real)
                fake_logits = self.discriminator(fake)
                loss_D_real = self.adv_loss(real_logits, valid)
                loss_D_fake = self.adv_loss(fake_logits, fake_labels)
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                self.opt_D.step()

                # === Train Generator ===
                self.opt_G.zero_grad(set_to_none=True)
                z = self._sample_noise(batch_size, seq_len)
                fake = self.generator(z)
                fake_logits = self.discriminator(fake)
                loss_G = self.adv_loss(fake_logits, valid)
                loss_G.backward()
                self.opt_G.step()

            if (epoch + 1) % max(1, num_epochs // 10) == 0:
                print(f"QuantGAN epoch {epoch + 1}/{num_epochs} | LossD: {loss_D.item():.4f} | LossG: {loss_G.item():.4f}")

        self.eval()
        self.generator.eval()
        self.discriminator.eval()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        generation_length: int,
        *args,
        **kwargs
    ) -> torch.Tensor:
        z = self._sample_noise(num_samples, generation_length)
        fake = self.generator(z)
        return fake.detach().cpu().squeeze(-1)