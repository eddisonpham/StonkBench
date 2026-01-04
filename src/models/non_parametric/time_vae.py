import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from src.models.base.base_model import DeepLearningModel

class Sampling(nn.Module):
    """Standard reparameterization for VAE."""
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, logvar_min, logvar_max):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h, _ = self.rnn(x)
        h_pool = torch.mean(h, dim=1)
        mu = self.fc_mu(h_pool)
        log_var = self.fc_log_var(h_pool)
        log_var = torch.clamp(log_var, min=self.logvar_min, max=self.logvar_max)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_dim, latent_dim, input_dim):
        super().__init__()
        self.seq_len = seq_len
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.decoder_rnn = nn.LSTM(
            input_size=hidden_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        z_expand = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        t = torch.arange(self.seq_len, device=z.device)
        pos = self.pos_emb(t).unsqueeze(0).repeat(z.size(0), 1, 1)
        dec_in = torch.cat([z_expand, pos], dim=-1)
        out, _ = self.decoder_rnn(dec_in)
        return self.output_layer(out)

class KLRamp:
    """Class to handle KL annealing schedule."""
    def __init__(self, kl_anneal_epochs, kl_weight_start, kl_weight_end):
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end

    def __call__(self, epoch: int):
        if self.kl_anneal_epochs < 1:
            return self.kl_weight_end
        e = max(0, epoch - 1)
        slope = (self.kl_weight_end - self.kl_weight_start) / float(self.kl_anneal_epochs)
        weight = self.kl_weight_start + slope * e
        return float(min(self.kl_weight_end, weight))

class VAELoss:
    """Class to compute the VAE loss, including reconstruction and KL terms."""
    def __init__(self, recon_weight=2.0, min_kl=0.0):
        self.recon_weight = recon_weight
        self.min_kl = min_kl

    def __call__(self, x, x_hat, mu, log_var, kl_weight, return_parts=False):
        recon_loss = nn.MSELoss(reduction='mean')(x_hat, x)
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        if self.min_kl > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.min_kl)
        kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
        loss = self.recon_weight * recon_loss + kl_weight * kl_loss
        if return_parts:
            return loss, recon_loss, kl_loss
        else:
            return loss

class TimeVAE(DeepLearningModel):
    """
    VAE for time series (univariate/multivariate).
    Implements slow KL ramp to avoid posterior collapse (KL ~ 0, spike output).
    """
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dim: int = 60,
        lr: float = 1e-3,
        kl_anneal_epochs: int = 100,
        kl_weight_start: float = 0.0,
        kl_weight_end: float = 1.0,
        recon_weight: float = 2.0,
        min_kl: float = 0.0,
        clip_grad: float = 2.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = torch.device(device)
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.clip_grad = clip_grad
        self.recon_weight = recon_weight
        self.min_kl = float(min_kl)

        self.LOGVAR_MIN = -8.0
        self.LOGVAR_MAX = 5.0

        # --- Modularized encoder and decoder ---
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, self.LOGVAR_MIN, self.LOGVAR_MAX)
        self.sampler = Sampling()
        self.decoder = Decoder(seq_len, hidden_dim, latent_dim, input_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        self.to(self.device)

        # KL Annealing and Loss
        self.kl_ramp = KLRamp(kl_anneal_epochs, kl_weight_start, kl_weight_end)
        self.loss_module = VAELoss(recon_weight, min_kl)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampler(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def fit(self, data_loader: DataLoader, num_epochs: int = 25, verbose: bool = True):
        """Train TimeVAE."""
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            kl_weight = self.kl_ramp(epoch)
            self.train()
            total_loss, total_recon, total_kl, num = 0.0, 0.0, 0.0, 0

            for batch_idx, batch in enumerate(data_loader, 1):
                batch_start_time = time.time()
                x = batch[0].float().to(self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                self.optimizer.zero_grad()
                x_hat, mu, log_var = self(x)
                loss, recon, kl = self.loss_module(x, x_hat, mu, log_var, kl_weight, return_parts=True)
                loss.backward()
                if self.clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
                self.optimizer.step()
                bsize = x.size(0)
                total_loss += loss.item() * bsize
                total_recon += recon.item() * bsize
                total_kl += kl.item() * bsize
                num += bsize
                batch_time = time.time() - batch_start_time
                if verbose:
                    print(f"  Epoch {epoch:03d} | Batch {batch_idx}/{len(data_loader)} | Train batch time: {batch_time:.2f}s")

            avg_loss = total_loss / num
            avg_recon = total_recon / num
            avg_kl = total_kl / num

            epoch_time = time.time() - epoch_start_time
            if verbose:
                print(f"Epoch {epoch:03d}: Train loss {avg_loss:.5g} (recon {avg_recon:.5g}, KL {avg_kl:.5g}) | Time: {epoch_time:.2f}s")
            
        if verbose:
            print(f"Training completed for {num_epochs} epochs.")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42, *args, **kwargs):
        """
        Generate synthetic samples after training.

        Args:
            num_samples (int): Number of simulated samples (R).
            generation_length (int): Length of each generated sample.
            seed (int, optional): Random seed for generation. Defaults to 42.
            *args, **kwargs: Optional arguments.

        Returns:
            torch.Tensor: Generated series of shape (R, l)
        """
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            x_hat = self.decoder(z)
            # x_hat shape: (num_samples, seq_len, input_dim)
            # Extract the requested length
            if x_hat.shape[1] > generation_length:
                x_hat = x_hat[:, :generation_length, :]
            elif x_hat.shape[1] < generation_length:
                pad_size = generation_length - x_hat.shape[1]
                pad = torch.zeros(x_hat.shape[0], pad_size, x_hat.shape[2], device=x_hat.device, dtype=x_hat.dtype)
                x_hat = torch.cat([x_hat, pad], dim=1)
            # Squeeze the last dimension to get (num_samples, generation_length) for univariate
            if x_hat.shape[2] == 1:
                x_hat = x_hat.squeeze(-1)
        return x_hat
