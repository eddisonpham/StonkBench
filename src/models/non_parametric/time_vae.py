import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC
from tqdm import tqdm

from src.models.base.base_model import DeepLearningModel


class TimeVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

class TimeVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_size, seq_len, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, z, seq_len=None):
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        h0 = torch.tanh(self.fc_init(z)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = torch.zeros_like(h0).to(z.device)
        lstm_input = h0[-1].unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(lstm_input, (h0, c0))
        return self.output_layer(out)

class TimeVAE(DeepLearningModel):
    """
    TimeVAE: Variational Autoencoder for OHLC Time Series Generation.

    Args:
        seq_len: Sequence length (l)
        num_features: Number of features (N)
        latent_dim: Dimension of latent vector
        hidden_dim: LSTM hidden dimension
        n_layers: Number of LSTM layers
        lr: Learning rate
    """
    def __init__(self, seq_len, num_features, latent_dim=16, hidden_dim=64, n_layers=1, lr=1e-3):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr

        self.encoder = TimeVAEEncoder(num_features, hidden_dim, latent_dim, n_layers).to(self.device)
        self.decoder = TimeVAEDecoder(latent_dim, hidden_dim, num_features, seq_len, n_layers).to(self.device)

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        self.trained = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def fit(self, data_loader, epochs=50, verbose=True):
        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.forward(batch)
                loss = self.loss_function(recon, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(data_loader):.6f}")
        self.trained = True

    @torch.no_grad()
    def generate(self, num_samples, seq_len=None):
        """
        Generate synthetic sequences.

        Args:
            num_samples (int): Number of samples (R)
            seq_len (int, optional): Sequence length (l). Defaults to training seq_len.

        Returns:
            torch.Tensor: Generated sequences (R, l, N)
        """
        if not self.trained:
            raise RuntimeError("TimeVAE must be trained before generation.")
        seq_len = seq_len or self.seq_len
        self.decoder.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.decoder(z, seq_len=seq_len).detach().cpu()
