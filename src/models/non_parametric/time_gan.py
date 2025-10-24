import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from src.models.base.base_model import DeepLearningModel


class RNNBlock(nn.Module):
    """Small helper: wrapper around GRU with optional layernorm and dropout."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)

    def forward(self, x, h0=None):
        out, h = self.gru(x, h0)
        return out, h


class Embedder(nn.Module):
    """Maps observed multivariate sequence to latent representation H"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = RNNBlock(input_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.act(self.fc(rnn_out))


class Recovery(nn.Module):
    """Maps latent representation H back to data space X"""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = RNNBlock(hidden_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        rnn_out, _ = self.rnn(h)
        return self.fc(rnn_out)


class Generator(nn.Module):
    """Generates latent space samples from random noise Z"""
    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = RNNBlock(noise_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, z):
        rnn_out, _ = self.rnn(z)
        return self.act(self.fc(rnn_out))


class Supervisor(nn.Module):
    """Supervisor network: predicts next-step latent dynamics (used for supervised loss)"""
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = RNNBlock(hidden_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, h):
        rnn_out, _ = self.rnn(h)
        return self.act(self.fc(rnn_out))


class Discriminator(nn.Module):
    """Discriminator that works in data space (X) or latent space (H)
    Accepts sequences and outputs a sequence of probabilities (or logits).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = RNNBlock(input_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out).squeeze(-1)


class TimeGAN(DeepLearningModel):
    """TimeGAN implementation.

    Example usage:
        model = TimeGAN(input_dim=N, hidden_dim=24)
        model.fit(train_loader, epochs=100)
        synth = model.generate(100)

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 24,
                 noise_dim: int = 32,
                 embedder_layers: int = 1,
                 generator_layers: int = 1,
                 supervisor_layers: int = 1,
                 discriminator_layers: int = 1,
                 lr: float = 1e-3,
                 device: Optional[torch.device] = None,
                 verbose: bool = True):
        super().__init__()
        if device is not None:
            self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.lr = lr
        self.verbose = verbose

        self.embedder = Embedder(input_dim, hidden_dim, num_layers=embedder_layers).to(self.device)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers=embedder_layers).to(self.device)

        self.generator = Generator(noise_dim, hidden_dim, num_layers=generator_layers).to(self.device)
        self.supervisor = Supervisor(hidden_dim, num_layers=supervisor_layers).to(self.device)

        self.discriminator = Discriminator(hidden_dim, hidden_dim, num_layers=discriminator_layers).to(self.device)

        all_params = list(self.embedder.parameters()) + list(self.recovery.parameters())
        self.opt_e = optim.Adam(all_params, lr=self.lr)
        self.opt_g = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=self.lr)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self._is_fitted = False

    def _sample_noise(self, batch_size: int, seq_len: int):
        return torch.rand(batch_size, seq_len, self.noise_dim, device=self.device) * 2 - 1

    @staticmethod
    def _get_seq_len_from_batch(batch):
        return batch.size(1)

    def _step_embedder_recovery(self, x):
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        loss_er = self.mse_loss(x_tilde, x)
        self.opt_e.zero_grad()
        loss_er.backward()
        self.opt_e.step()
        return loss_er.item()

    def _step_supervised(self, x):
        h = self.embedder(x).detach()
        h_hat_super = self.supervisor(h)
        loss_s = self.mse_loss(h_hat_super[:, :-1, :], h[:, 1:, :])
        self.opt_g.zero_grad()
        loss_s.backward()
        self.opt_g.step()
        return loss_s.item()

    def _step_discriminator(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_real = self.embedder(x).detach()

        z = self._sample_noise(batch_size, seq_len)
        e_fake = self.generator(z)
        h_fake = self.supervisor(e_fake).detach()

        logits_real = self.discriminator(h_real)
        logits_fake = self.discriminator(h_fake)

        y_real = torch.ones_like(logits_real, device=self.device)
        y_fake = torch.zeros_like(logits_fake, device=self.device)

        loss_real = self.bce_loss(logits_real, y_real)
        loss_fake = self.bce_loss(logits_fake, y_fake)
        loss_d = loss_real + loss_fake

        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        return loss_d.item()

    def _step_generator(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        z = self._sample_noise(batch_size, seq_len)
        e_fake = self.generator(z)
        h_fake = self.supervisor(e_fake)
        logits_fake = self.discriminator(h_fake)
        y_real = torch.ones_like(logits_fake, device=self.device)
        gadv_loss = self.bce_loss(logits_fake, y_real)
        g_supervised_loss = self.mse_loss(h_fake[:, :-1, :], h_fake[:, 1:, :])
        x_fake = self.recovery(h_fake)
        x_real = x
        loss_moments = torch.mean((torch.mean(x_fake, dim=0) - torch.mean(x_real, dim=0)) ** 2)

        gen_loss = gadv_loss + 100 * g_supervised_loss + 100 * loss_moments

        self.opt_g.zero_grad()
        gen_loss.backward()
        self.opt_g.step()

        return {
            'g_adv': gadv_loss.item(),
            'g_sup': g_supervised_loss.item(),
            'g_mom': loss_moments.item(),
            'g_total': gen_loss.item(),
        }

    def fit(self, data_loader, epochs: int = 100, embedder_steps: int = 5, supervisor_steps: int = 5, adversarial_steps: int = 100):
        """Train TimeGAN using the common 3-phase routine:
        1) Train embedder + recovery (reconstruction)
        2) Train supervisor to capture dynamics (supervised loss)
        3) Joint adversarial training (discriminator vs generator+supervisor) while optionally fine-tuning embedder/recovery

        Args:
            data_loader (DataLoader): yields batches of (B, T, N)
            epochs (int): number of outer epochs
            embedder_steps (int): how many steps of ER pretrain per epoch
            supervisor_steps (int): how many steps of supervisor pretrain per epoch
            adversarial_steps (int): how many adversarial minibatches per epoch
        """
        self.train()
        for epoch in range(epochs):
            epoch_er_losses = []
            epoch_s_losses = []
            epoch_d_losses = []
            epoch_g_losses = []

            # Phase 1: embedder + recovery
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} — Phase 1: ER pretrain")
            for _ in range(embedder_steps):
                for batch in data_loader:
                    x = batch.to(self.device).float()
                    loss_er = self._step_embedder_recovery(x)
                    epoch_er_losses.append(loss_er)

            # Phase 2: supervisor
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} — Phase 2: Supervisor pretrain")
            for _ in range(supervisor_steps):
                for batch in data_loader:
                    x = batch.to(self.device).float()
                    loss_s = self._step_supervised(x)
                    epoch_s_losses.append(loss_s)

            # Phase 3: adversarial
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} — Phase 3: Adversarial training")

            data_iter = iter(data_loader)
            for it in range(adversarial_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)
                x = batch.to(self.device).float()
                d_loss = self._step_discriminator(x)
                g_losses = self._step_generator(x)
                epoch_d_losses.append(d_loss)
                epoch_g_losses.append(g_losses['g_total'])

            if self.verbose:
                print(f"Epoch {epoch+1} summary: ER={sum(epoch_er_losses)/len(epoch_er_losses) if epoch_er_losses else 0:.4f}, S={sum(epoch_s_losses)/len(epoch_s_losses) if epoch_s_losses else 0:.4f}, D={sum(epoch_d_losses)/len(epoch_d_losses) if epoch_d_losses else 0:.4f}, G={sum(epoch_g_losses)/len(epoch_g_losses) if epoch_g_losses else 0:.4f}")

        self._is_fitted = True

    def generate(self, num_samples: int, seq_len: int = None) -> torch.Tensor:
        """Generate num_samples of synthetic sequences of length seq_len. If seq_len is None, raises an error
        unless a default was established during training (not tracked here).

        Returns:
            torch.Tensor: (num_samples, seq_len, input_dim)
        """
        if not self._is_fitted:
            print("Warning: generate() called before fit(); results may be untrained.")

        if seq_len is None:
            raise ValueError("seq_len must be provided to generate sequences.")

        self.eval()
        with torch.no_grad():
            z = self._sample_noise(num_samples, seq_len)
            e_fake = self.generator(z)
            h_fake = self.supervisor(e_fake)
            x_fake = self.recovery(h_fake)
        return x_fake.cpu()


