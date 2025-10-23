# timevae_pytorch.py
import math
from abc import ABC
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from src.models.base.base_model import DeepLearningModel


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

# -------------------------
# Encoder
# -------------------------
class TemporalEncoder(nn.Module):
    """
    Conv1D encoder that maps (batch, L, N) -> global summary -> (mu, logvar)
    Input expected shape: (batch, seq_length, n_channels)
    """
    def __init__(self, seq_length: int, n_channels: int, hidden_channels: int = 128,
                 latent_dim: int = 32, use_lstm: bool = True):
        super().__init__()
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.use_lstm = use_lstm

        # Conv1d expects (batch, channels, length) -> we permute in forward
        self.conv1 = nn.Conv1d(n_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.act = nn.ReLU(inplace=True)

        if use_lstm:
            # bidirectional LSTM with hidden/2 in each direction -> outputs hidden_channels features
            self.lstm = nn.LSTM(hidden_channels, hidden_channels // 2,
                                num_layers=1, batch_first=True, bidirectional=True)
            agg_dim = hidden_channels
        else:
            agg_dim = hidden_channels

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1) -> squeeze -> (B, C)
        self.fc_mu = nn.Linear(agg_dim, latent_dim)
        self.fc_logvar = nn.Linear(agg_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, L, N)
        x = x.permute(0, 2, 1)                 # -> (B, N, L) for Conv1d
        x = self.act(self.bn1(self.conv1(x)))  # (B, hidden, L)
        x = self.act(self.bn2(self.conv2(x)))  # (B, hidden, L)

        if self.use_lstm:
            x_l = x.permute(0, 2, 1)           # -> (B, L, hidden) for LSTM (batch_first=True)
            out, _ = self.lstm(x_l)            # out: (B, L, hidden)
            agg = out.mean(dim=1)              # aggregate over time -> (B, hidden)
        else:
            agg = self.pool(x).squeeze(-1)     # (B, hidden)

        mu = self.fc_mu(agg)
        logvar = self.fc_logvar(agg)
        return mu, logvar

# -------------------------
# Decoder
# -------------------------
class TemporalDecoder(nn.Module):
    """
    Decoder: latent z -> (batch, L, N)
    Projects z to a temporal feature map and upsamples to seq_length, then conv -> n_channels
    """
    def __init__(self, seq_length: int, n_channels: int, latent_dim: int = 32,
                 hidden_channels: int = 128, n_resblocks: int = 2):
        super().__init__()
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # initial temporal length (small) to expand from
        self.init_time = max(4, seq_length // 8)
        self.fc = nn.Linear(latent_dim, hidden_channels * self.init_time)

        # build upsample chain to reach seq_length
        self.upsample_blocks = nn.ModuleList()
        cur_len = self.init_time
        cur_ch = hidden_channels

        # keep doubling length until we reach at least seq_length
        while cur_len < seq_length:
            next_ch = max(cur_ch // 2, n_channels * 2)
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(cur_ch, next_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(next_ch),
                nn.ReLU(inplace=True)
            )
            self.upsample_blocks.append(block)
            cur_ch = next_ch
            cur_len = cur_len * 2
            # safety: break if we've overshot by a lot
            if cur_len > seq_length * 4:
                break

        # some residual convs at the final resolution
        res = []
        for _ in range(n_resblocks):
            res.append(nn.Conv1d(cur_ch, cur_ch, kernel_size=3, padding=1))
            res.append(nn.ReLU(inplace=True))
        self.resnet = nn.Sequential(*res) if res else nn.Identity()

        self.final_conv = nn.Conv1d(cur_ch, n_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, seq_length: Optional[int] = None):
        """
        z: (B, latent_dim)
        seq_length: if provided, will interpolate final temporal dim to this length
        returns (B, seq_length, n_channels)
        """
        if seq_length is None:
            seq_length = self.seq_length

        batch = z.size(0)
        x = self.fc(z)                              # (B, hidden * init_time)
        x = x.view(batch, self.hidden_channels, self.init_time)  # (B, C, T0)

        for up in self.upsample_blocks:
            x = up(x)                               # (B, C', T doubled)

        x = self.resnet(x)                          # (B, cur_ch, T')
        if x.size(-1) != seq_length:
            # linear interpolation along temporal axis (expects (B, C, L))
            x = F.interpolate(x, size=seq_length, mode="linear", align_corners=False)

        x = self.final_conv(x)                      # (B, n_channels, seq_length)
        x = x.permute(0, 2, 1).contiguous()         # -> (B, seq_length, n_channels)
        return x

# -------------------------
# TimeVAE (standalone)
# -------------------------
class TimeVAE(nn.Module):
    """
    VAE for multivariate time series.
    - inputs: (B, seq_length, n_channels)
    - generate(num_samples, seq_length=...) -> (num_samples, seq_length, n_channels)
    """
    def __init__(self, seq_length: int, n_channels: int, latent_dim: int = 32,
                 enc_hidden: int = 128, dec_hidden: int = 128, beta: float = 1.0,
                 use_lstm: bool = True, device: Optional[torch.device] = None):
        super().__init__()
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.beta = beta

        # device fallback if not provided (works standalone)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = TemporalEncoder(seq_length, n_channels, hidden_channels=enc_hidden,
                                       latent_dim=latent_dim, use_lstm=use_lstm)
        self.decoder = TemporalDecoder(seq_length, n_channels, latent_dim=latent_dim,
                                       hidden_channels=dec_hidden)

    def to(self, *args, **kwargs):
        # keep track of device when model.to(device) called
        m = super().to(*args, **kwargs)
        try:
            # infer device from parameters
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return m

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, seq_length=self.seq_length)
        return recon, mu, logvar

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld, recon_loss, kld

    def fit(self,
            data_loader: DataLoader,
            epochs: int = 100,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            verbose: bool = True):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kld = 0.0
            n_batches = 0
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device).float()
                optimizer.zero_grad()
                recon, mu, logvar = self.forward(x)
                loss, recon_l, kld_l = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_recon += float(recon_l.item())
                epoch_kld += float(kld_l.item())
                n_batches += 1

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(f"Epoch {epoch}/{epochs} | loss: {epoch_loss / n_batches:.6f} | recon: {epoch_recon / n_batches:.6f} | kld: {epoch_kld / n_batches:.6f}")

    @torch.no_grad()
    def generate(self, num_samples: int, seq_length: Optional[int] = None, z: Optional[torch.Tensor] = None):
        """
        Generate sequences.
        - If z is None: sample z ~ N(0, I)
        - z shape: (num_samples, latent_dim) or (latent_dim,)
        Returns: torch.Tensor on CPU with shape (num_samples, seq_length, n_channels)
        """
        self.eval()
        seq_length = seq_length if seq_length is not None else self.seq_length

        if z is None:
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
        else:
            z = z.to(self.device).float()
            if z.dim() == 1:
                z = z.unsqueeze(0)
            if z.size(0) != num_samples:
                # allow generating for the provided z (override num_samples)
                num_samples = z.size(0)

        out = self.decoder(z, seq_length=seq_length)  # (B, seq_length, n_channels)
        return out.cpu()
