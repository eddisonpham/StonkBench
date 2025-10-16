import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.models.base.base_model import DeepLearningModel


class Embedder(nn.Module):
    """Maps real sequences to latent space."""
    def __init__(self, N, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=N - 1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)


class Recovery(nn.Module):
    """Recovers sequences from latent space."""
    def __init__(self, hidden_dim, N):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, N - 1)

    def forward(self, h):
        h, _ = self.gru(h)
        return self.fc(h)


class Generator(nn.Module):
    """Generates latent sequences from noise and time gaps."""
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim + 1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, delta_t):
        inp = torch.cat([z, delta_t], dim=-1)
        h, _ = self.gru(inp)
        return self.fc(h)


class Supervisor(nn.Module):
    """Supervises generator to capture temporal dynamics."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        h, _ = self.gru(h)
        return self.fc(h)


class Discriminator(nn.Module):
    """Distinguishes real vs fake sequences in latent space."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim + 1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, delta_t):
        inp = torch.cat([h, delta_t], dim=-1)
        h, _ = self.gru(inp)
        out = self.sigmoid(self.fc(h))  # sequence-wise output
        return out.mean(dim=1)  # average over sequence length


class TimeGAN(DeepLearningModel):
    """
    TimeGAN implementation for unevenly spaced multivariate time series.
    Input: (batch_size, l, N), 0-th channel = timestamps.
    """

    def __init__(self, l, N, latent_dim=32, hidden_dim=64, lr=2e-4, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.embedder = Embedder(N, hidden_dim).to(self.device)
        self.recovery = Recovery(hidden_dim, N).to(self.device)
        self.generator = Generator(latent_dim, hidden_dim).to(self.device)
        self.supervisor = Supervisor(hidden_dim).to(self.device)
        self.discriminator = Discriminator(hidden_dim).to(self.device)

        # Optimizers
        self.opt_E = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.l = l
        self.N = N
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.loss_fn = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.mean_dt = 1.0

    def _compute_delta_t(self, timestamps):
        delta_t = timestamps[:, 1:, :] - timestamps[:, :-1, :]
        delta_t = torch.cat([torch.zeros_like(delta_t[:, :1, :]), delta_t], dim=1)
        # Normalize per batch for stability
        delta_t = delta_t / (delta_t.max(dim=1, keepdim=True)[0] + 1e-8)
        return delta_t

    def fit(self, data_loader: DataLoader, epochs=100):
        all_dts = []

        for epoch in range(epochs):
            e_losses, g_losses, d_losses = [], [], []

            for real_data in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data = real_data.to(self.device)
                timestamps = real_data[..., [0]]
                features = real_data[..., 1:]
                delta_t = self._compute_delta_t(timestamps)
                all_dts.append(delta_t.mean().item())
                batch_size = real_data.size(0)

                # === Train Embedder / Recovery (autoencoder) ===
                self.opt_E.zero_grad()
                H = self.embedder(features)
                X_tilde = self.recovery(H)
                loss_E = self.mse_loss(X_tilde, features)
                loss_E.backward()
                self.opt_E.step()
                e_losses.append(loss_E.item())

                # === Train Generator + Supervisor ===
                self.opt_G.zero_grad()
                Z = torch.randn(batch_size, self.l, self.latent_dim, device=self.device)
                H_fake = self.generator(Z, delta_t)
                H_fake_sup = self.supervisor(H_fake)
                # Supervised loss: predicted next latent state
                loss_S = self.mse_loss(H_fake_sup[:, :-1, :], H_fake[:, 1:, :])

                # Adversarial loss for generator
                D_fake = self.discriminator(H_fake_sup, delta_t)
                g_loss_adv = self.loss_fn(D_fake, torch.ones_like(D_fake))
                loss_G = loss_S + g_loss_adv

                loss_G.backward()
                self.opt_G.step()
                g_losses.append(loss_G.item())

                # === Train Discriminator ===
                self.opt_D.zero_grad()
                H_real = self.embedder(features).detach()
                y_real = torch.ones(batch_size, 1, device=self.device)
                y_fake = torch.zeros(batch_size, 1, device=self.device)
                D_real = self.discriminator(H_real, delta_t)
                D_fake = self.discriminator(H_fake_sup.detach(), delta_t)
                loss_D = self.loss_fn(D_real, y_real) + self.loss_fn(D_fake, y_fake)
                loss_D.backward()
                self.opt_D.step()
                d_losses.append(loss_D.item())

            print(f"Epoch [{epoch+1}/{epochs}]  E_loss: {np.mean(e_losses):.4f}  G_loss: {np.mean(g_losses):.4f}  D_loss: {np.mean(d_losses):.4f}")

        self.mean_dt = np.mean(all_dts)
        print(f"Average normalized Δt from training: {self.mean_dt:.4f}")

    def generate(self, num_samples, timestamps=None, linear_timestamps=False, future_length=None):
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        L = future_length or self.l
        with torch.no_grad():
            if timestamps is not None:
                delta_t = self._compute_delta_t(timestamps.to(self.device))
                timestamps = timestamps.to(self.device)
            elif linear_timestamps:
                timestamps = torch.linspace(0, 1, L, device=self.device).unsqueeze(0).unsqueeze(-1)
                timestamps = timestamps.repeat(num_samples, 1, 1)
                delta_t = self._compute_delta_t(timestamps)
            else:
                delta_t = torch.abs(torch.randn(num_samples, L, 1, device=self.device) * self.mean_dt)
                timestamps = torch.cumsum(delta_t, dim=1)

            Z = torch.randn(num_samples, L, self.latent_dim, device=self.device)
            H_fake = self.generator(Z, delta_t)
            H_fake_sup = self.supervisor(H_fake)
            X_fake = self.recovery(H_fake_sup)
            fake_series = torch.cat([timestamps, X_fake], dim=-1)

        return fake_series.cpu().numpy()
