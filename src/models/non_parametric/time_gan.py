"""
PyTorch implementation of TimeGAN.

This module defines a TimeGAN model that inherits from DeepLearningModel
and follows the formatting/style used by parametric models. It removes data
normalization and suppresses print outputs.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base.base_model import DeepLearningModel


def _make_rnn(input_dim: int, hidden_dim: int, num_layers: int, kind: str) -> nn.Module:
    kind = (kind or "gru").lower()
    if kind == "lstm":
        return nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    if kind == "rnn":
        return nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    return nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)


class _Embedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, rnn_kind: str):
        super().__init__()
        self.rnn = _make_rnn(input_dim, hidden_dim, num_layers, rnn_kind)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.proj(outputs)


class _Recovery(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, rnn_kind: str):
        super().__init__()
        self.rnn = _make_rnn(hidden_dim, hidden_dim, num_layers, rnn_kind)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(h)
        return self.proj(outputs)


class _Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int, rnn_kind: str):
        super().__init__()
        self.rnn = _make_rnn(z_dim, hidden_dim, num_layers, rnn_kind)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(z)
        return self.proj(outputs)


class _Supervisor(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, rnn_kind: str):
        super().__init__()
        num_layers = max(1, num_layers - 1)
        self.rnn = _make_rnn(hidden_dim, hidden_dim, num_layers, rnn_kind)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(h)
        return self.proj(outputs)


class _Discriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, rnn_kind: str):
        super().__init__()
        self.rnn = _make_rnn(hidden_dim, hidden_dim, num_layers, rnn_kind)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(h)
        return self.head(outputs)


class TimeGAN(DeepLearningModel):
    """
    TimeGAN implemented with PyTorch.

    - Inherits from DeepLearningModel and follows its interface.
    - Expects fixed-length sequences from the provided DataLoader in fit.
    - Does not normalize data and does not print training logs.
    """

    def __init__(
        self,
        length: int,
        num_channels: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        rnn_kind: str = "gru",
        gamma: float = 1.0,
    ):
        super().__init__(length=length, num_channels=num_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn_kind = str(rnn_kind)
        self.gamma = float(gamma)

        # Networks
        self.embedder = _Embedder(num_channels, hidden_dim, num_layers, rnn_kind)
        self.recovery = _Recovery(hidden_dim, num_channels, num_layers, rnn_kind)
        self.generator = _Generator(num_channels, hidden_dim, num_layers, rnn_kind)
        self.supervisor = _Supervisor(hidden_dim, num_layers, rnn_kind)
        self.discriminator = _Discriminator(hidden_dim, num_layers, rnn_kind)

        self.to(self.device)

    def _moment_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mean_real = x.mean(dim=0)
        mean_fake = x_hat.mean(dim=0)
        std_real = x.std(dim=0).clamp_min(1e-6)
        std_fake = x_hat.std(dim=0).clamp_min(1e-6)
        return (mean_fake - mean_real).abs().mean() + (std_fake - std_real).abs().mean()

    def _sample_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.num_channels, device=self.device)

    def fit(
        self,
        data_loader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        bce_logits = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        # Optimizers
        opt_er = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr, betas=(beta1, beta2))
        opt_gs = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr, betas=(beta1, beta2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        self.train()

        for epoch in range(num_epochs):
            for real_batch in data_loader:
                if isinstance(real_batch, (list, tuple)):
                    real_batch = real_batch[0]
                x = real_batch.to(self.device)  # (B, L, N)
                batch_size, seq_len, _ = x.shape

                # 1) Embedder/Recovery autoencoder
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                e_loss_t0 = mse(x_tilde, x)
                e_loss = 10.0 * torch.sqrt(e_loss_t0.clamp_min(1e-12))
                opt_er.zero_grad(set_to_none=True)
                e_loss.backward()
                opt_er.step()

                # 2) Supervisor loss (predict next step in latent space)
                with torch.no_grad():
                    h_detached = self.embedder(x)
                h_hat_supervise = self.supervisor(h_detached)
                g_loss_s = mse(h_detached[:, 1:, :], h_hat_supervise[:, :-1, :])
                opt_gs.zero_grad(set_to_none=True)
                g_loss_s.backward()
                opt_gs.step()

                # 3) Joint training
                for _ in range(2):  # Generator updates more than D
                    z = self._sample_noise(batch_size, seq_len)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    x_hat = self.recovery(h_hat)

                    # Adversarial losses
                    y_fake = self.discriminator(h_hat)
                    y_fake_e = self.discriminator(e_hat)
                    ones = torch.ones_like(y_fake)
                    g_loss_u = bce_logits(y_fake, ones)
                    g_loss_u_e = bce_logits(y_fake_e, torch.ones_like(y_fake_e))

                    # Moment loss between x_hat and x
                    g_loss_v = self._moment_loss(x_hat, x)

                    # Supervised loss again to stabilize
                    with torch.no_grad():
                        h_sup = self.embedder(x)
                    h_sup_pred = self.supervisor(h_sup)
                    g_loss_s2 = mse(h_sup[:, 1:, :], h_sup_pred[:, :-1, :])

                    g_loss = g_loss_u + self.gamma * g_loss_u_e + 100.0 * torch.sqrt(g_loss_s2.clamp_min(1e-12)) + 100.0 * g_loss_v

                    opt_gs.zero_grad(set_to_none=True)
                    g_loss.backward()
                    opt_gs.step()

                    # Update embedder slightly via reconstruction
                    h_new = self.embedder(x)
                    x_tilde_new = self.recovery(h_new)
                    e_loss_t0_new = mse(x_tilde_new, x)
                    e_loss_total = 10.0 * torch.sqrt(e_loss_t0_new.clamp_min(1e-12)) + 0.1 * g_loss_s2.detach()
                    opt_er.zero_grad(set_to_none=True)
                    e_loss_total.backward()
                    opt_er.step()

                # 4) Discriminator update
                with torch.no_grad():
                    z = self._sample_noise(batch_size, seq_len)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    h_real = self.embedder(x)

                y_real = self.discriminator(h_real)
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)

                d_loss_real = bce_logits(y_real, torch.ones_like(y_real))
                d_loss_fake = bce_logits(y_fake, torch.zeros_like(y_fake))
                d_loss_fake_e = bce_logits(y_fake_e, torch.zeros_like(y_fake_e))
                d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            print(f"TimeGAN epoch {epoch + 1}/{num_epochs}")

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
        z = torch.randn(num_samples, seq_length, self.num_channels, device=self.device)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        x_hat = self.recovery(h_hat)
        return x_hat.detach().cpu()