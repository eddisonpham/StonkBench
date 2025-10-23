import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.base.base_model import DeepLearningModel



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.sigmoid(self.fc(out))
        return out


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.sigmoid(self.fc(out))
        return out


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Embedder, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers)

    def forward(self, x):
        return self.model(x)

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers):
        super(Recovery, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, output_dim, n_layers)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Generator, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers)

    def forward(self, x):
        return self.model(x)

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(Supervisor, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, hidden_dim, n_layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(Discriminator, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, 1, n_layers)

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange

from src.models.base.base_model import DeepLearningModel

# ============================================================
# DeepLearningModel-compatible TimeGAN
# ============================================================

class TimeGAN(DeepLearningModel):
    def __init__(
        self,
        seq_len,
        feature_dim,
        hidden_dim,
        n_layers,
        n_seq,
        gamma=1.0,
        batch_size=128,
        lr=1e-3,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_seq = n_seq
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr

        # ------------------- Modules -------------------
        self.embedder = Embedder(feature_dim, hidden_dim, n_layers).to(self.device)
        self.recovery = Recovery(hidden_dim, feature_dim, n_layers).to(self.device)
        self.generator = Generator(n_seq, hidden_dim, n_layers).to(self.device)
        self.supervisor = Supervisor(hidden_dim, n_layers).to(self.device)
        self.discriminator = Discriminator(hidden_dim, n_layers).to(self.device)
        self.autoencoder = nn.Sequential(self.embedder, self.recovery).to(self.device)

        # ------------------- Optimizers -------------------
        self.opt_embedder = optim.Adam(self.embedder.parameters(), lr=lr)
        self.opt_recovery = optim.Adam(self.recovery.parameters(), lr=lr)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)

        # ------------------- Losses -------------------
        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()

    # ============================================================
    # Public API
    # ============================================================

    def fit(self, data_loader: DataLoader, num_epochs: int = 10):
        """Unified training loop for all phases."""
        for epoch in range(num_epochs):
            for x in data_loader:
                x = x.to(self.device).float()

                # --- Phase 1: Autoencoder ---
                self._train_autoencoder(x)

                # --- Phase 2: Supervisor ---
                self._train_supervisor(x)

                # --- Phase 3: Adversarial training ---
                Z = self._sample_noise(x.size(0))
                self._train_generator(x, Z)
                self._train_discriminator(x, Z)

            if num_epochs % 10 == 0:
                print(f"[Epoch {num_epochs}] Training step completed.")

    def generate(self, num_samples: int):
        """Generate synthetic sequences."""
        with torch.no_grad():
            Z = torch.rand(num_samples, self.seq_len, self.n_seq, device=self.device)
            H_hat = self.supervisor(self.generator(Z))
            X_hat = self.recovery(H_hat)
        return X_hat.cpu()

    # ============================================================
    # Internal modular training functions
    # ============================================================

    def _train_autoencoder(self, x):
        self.opt_embedder.zero_grad()
        self.opt_recovery.zero_grad()
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        loss_t0 = self.criterion_mse(x, x_tilde)
        e_loss = 10 * torch.sqrt(loss_t0)
        e_loss.backward()
        self.opt_embedder.step()
        self.opt_recovery.step()
        return e_loss.item()

    def _train_supervisor(self, x):
        self.opt_supervisor.zero_grad()
        h = self.embedder(x)
        h_hat = self.supervisor(h)
        loss_s = self.criterion_mse(h[:, 1:, :], h_hat[:, 1:, :])
        loss_s.backward()
        self.opt_supervisor.step()
        return loss_s.item()

    def _train_generator(self, x, z):
        self.opt_generator.zero_grad()
        self.opt_supervisor.zero_grad()

        y_fake = self.discriminator(self.supervisor(self.generator(z)))
        loss_u = self.criterion_bce(y_fake, torch.ones_like(y_fake))

        y_fake_e = self.discriminator(self.generator(z))
        loss_u_e = self.criterion_bce(y_fake_e, torch.ones_like(y_fake_e))

        h = self.embedder(x)
        h_hat_supervised = self.supervisor(h)
        loss_s = self.criterion_mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_hat = self.recovery(self.supervisor(self.generator(z)))
        loss_moments = self._generator_moments_loss(x, x_hat)

        total_loss = loss_u + loss_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_moments
        total_loss.backward()
        self.opt_generator.step()
        self.opt_supervisor.step()
        return total_loss.item()

    def _train_discriminator(self, x, z):
        self.opt_discriminator.zero_grad()
        loss = self._discriminator_loss(x, z)
        loss.backward()
        self.opt_discriminator.step()
        return loss.item()

    # ============================================================
    # Loss helpers
    # ============================================================

    def _discriminator_loss(self, x, z):
        y_real = self.discriminator(self.embedder(x))
        loss_real = self.criterion_bce(y_real, torch.ones_like(y_real))

        y_fake = self.discriminator(self.supervisor(self.generator(z)))
        loss_fake = self.criterion_bce(y_fake, torch.zeros_like(y_fake))

        y_fake_e = self.discriminator(self.generator(z))
        loss_fake_e = self.criterion_bce(y_fake_e, torch.zeros_like(y_fake_e))

        return loss_real + loss_fake + self.gamma * loss_fake_e

    @staticmethod
    def _generator_moments_loss(x_true, x_pred):
        mean_true, var_true = torch.mean(x_true, dim=0), torch.var(x_true, dim=0)
        mean_pred, var_pred = torch.mean(x_pred, dim=0), torch.var(x_pred, dim=0)
        loss_mean = torch.mean(torch.abs(mean_true - mean_pred))
        loss_var = torch.mean(torch.abs(torch.sqrt(var_true + 1e-6) - torch.sqrt(var_pred + 1e-6)))
        return loss_mean + loss_var

    # ============================================================
    # Noise helper
    # ============================================================

    def _sample_noise(self, batch_size):
        return torch.rand(batch_size, self.seq_len, self.n_seq, device=self.device).float()
