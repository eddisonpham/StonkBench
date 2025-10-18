import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.models.base.base_model import DeepLearningModel


class Embedder(nn.Module):
    """Maps real sequences to latent space.
    
    Assumes input shape (batch_size, seq_length, feature_dim).
    Note: Feature dimension does not include timestamps channel.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim, 
                         hidden_size=hidden_dim, 
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, feature_dim)
        
        Returns:
            Embedded representation of shape (batch_size, seq_length, hidden_dim)
        """
        h, _ = self.gru(x)
        return self.fc(h)


class Recovery(nn.Module):
    """Recovers sequences from latent space.
    
    Maps from hidden representation back to original feature space.
    """
    def __init__(self, hidden_dim, feature_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, 
                         hidden_size=hidden_dim, 
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, h):
        """
        Args:
            h: Latent representation of shape (batch_size, seq_length, hidden_dim)
            
        Returns:
            Reconstructed features of shape (batch_size, seq_length, feature_dim)
        """
        h, _ = self.gru(h)
        return self.fc(h)


class Generator(nn.Module):
    """Generates latent sequences from noise and time stamps.
    
    Takes random noise and timestamp information to generate synthetic
    sequences in the latent space.
    """
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim + 1,  # +1 for delta_t
                         hidden_size=hidden_dim, 
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, z, delta_t):
        """
        Args:
            z: Random noise of shape (batch_size, seq_length, latent_dim)
            delta_t: Time differences of shape (batch_size, seq_length, 1)
            
        Returns:
            Generated sequences in latent space (batch_size, seq_length, hidden_dim)
        """
        # Concatenate noise and time information along feature dimension
        inp = torch.cat([z, delta_t], dim=-1)
        h, _ = self.gru(inp)
        return self.fc(h)


class Supervisor(nn.Module):
    """Supervises generator to capture temporal dynamics.
    
    Helps the generator learn the sequential dependencies in the data.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, 
                         hidden_size=hidden_dim, 
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, h):
        """
        Args:
            h: Latent representation of shape (batch_size, seq_length, hidden_dim)
            
        Returns:
            Supervised representation of shape (batch_size, seq_length, hidden_dim)
        """
        h, _ = self.gru(h)
        return self.fc(h)


class Discriminator(nn.Module):
    """Distinguishes real vs fake sequences in latent space.
    
    Takes sequences in latent space along with time information to
    determine if they are real or synthetic.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim + 1,  # +1 for delta_t
                         hidden_size=hidden_dim, 
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h, delta_t):
        """
        Args:
            h: Latent representation of shape (batch_size, seq_length, hidden_dim)
            delta_t: Time differences of shape (batch_size, seq_length, 1)
            
        Returns:
            Classification scores of shape (batch_size, 1)
        """
        # Concatenate latent representation and time information
        inp = torch.cat([h, delta_t], dim=-1)
        h, _ = self.gru(inp)
        out = self.sigmoid(self.fc(h))  # sequence-wise output
        return out.mean(dim=1)  # average over sequence length


class TimeGAN(DeepLearningModel):
    """
    TimeGAN implementation for unevenly spaced multivariate time series.
    
    This implementation supports generating synthetic time series data from unevenly
    spaced multivariate time series. The model assumes the 0-th input channel contains
    unevenly spaced timestamps.
    
    Assumptions:
      - Input arrays shaped (batch_size, seq_length, channels) where channel 0 is timestamp,
        channels 1..N-1 are the feature signals (e.g., OHLC: Open, Close, High, Low).
      - Data is typically normalized to [0, 1] range for stable training.
      - Generates synthetic time series of shape (num_samples, seq_length, channels).
    """
    
    def __init__(self, seq_length, num_channels, latent_dim=32, hidden_dim=64, lr=2e-4, device=None):
        """
        Args:
            seq_length: Length of each time series sequence
            num_channels: Number of channels in the input data (including timestamp channel)
            latent_dim: Dimension of the latent space for the generator
            hidden_dim: Hidden dimension for GRU cells
            lr: Learning rate for the optimizers
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature dimension is all channels except timestamp
        self.feature_dim = num_channels - 1
        
        # Networks
        self.embedder = Embedder(self.feature_dim, hidden_dim).to(self.device)
        self.recovery = Recovery(hidden_dim, self.feature_dim).to(self.device)
        self.generator = Generator(latent_dim, hidden_dim).to(self.device)
        self.supervisor = Supervisor(hidden_dim).to(self.device)
        self.discriminator = Discriminator(hidden_dim).to(self.device)
        
        # Optimizers
        self.opt_E = optim.Adam(list(self.embedder.parameters()) + 
                               list(self.recovery.parameters()), 
                               lr=lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(list(self.generator.parameters()) + 
                               list(self.supervisor.parameters()), 
                               lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.discriminator.parameters(), 
                               lr=lr, betas=(0.5, 0.999))
        
        # Model parameters
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.loss_fn = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.mean_dt = 1.0
        
        # Storage for generation statistics
        self.mean_initial_values = None
    
    def _compute_delta_t(self, timestamps):
        """
        Compute time differences between adjacent timestamps and normalize them.
        
        Args:
            timestamps: Tensor of shape (batch_size, seq_length, 1) containing timestamps
            
        Returns:
            Normalized time differences of shape (batch_size, seq_length, 1)
        """
        # Compute differences between adjacent timestamps
        delta_t = timestamps[:, 1:, :] - timestamps[:, :-1, :]
        # Add zero as the first difference (no previous timestamp for the first one)
        delta_t = torch.cat([torch.zeros_like(delta_t[:, :1, :]), delta_t], dim=1)
        # Normalize by the maximum difference in each batch for training stability
        delta_t = delta_t / (delta_t.max(dim=1, keepdim=True)[0] + 1e-8)
        return delta_t
    
    def fit(self, data_loader: DataLoader, epochs=1):
        """
        Train the TimeGAN model on batched time series data.
        
        Args:
            data_loader: DataLoader yielding batches of shape (batch_size, seq_length, channels)
                         where the first channel is timestamps
            epochs: Number of training epochs
        """
        all_dts = []
        all_initial_values = []
        
        for epoch in range(epochs):
            e_losses, g_losses, d_losses = [], [], []
            
            for real_data in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data = real_data.to(self.device)
                
                # Split timestamps and features
                timestamps = real_data[..., [0]]  # Shape: (batch_size, seq_length, 1)
                features = real_data[..., 1:]     # Shape: (batch_size, seq_length, feature_dim)
                
                # Compute time differences
                delta_t = self._compute_delta_t(timestamps)
                all_dts.append(delta_t.mean().item())
                
                # Store initial feature values from training data (first timestep)
                if epoch == 0:  # Only collect on first epoch
                    all_initial_values.append(features[:, 0, :].cpu())
                
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
                Z = torch.randn(batch_size, self.seq_length, self.latent_dim, device=self.device)
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
        
        # Compute mean initial values across all training samples
        if len(all_initial_values) > 0:
            all_initial_values = torch.cat(all_initial_values, dim=0)  # (total_samples, feature_dim)
            self.mean_initial_values = all_initial_values.mean(dim=0).numpy()  # (feature_dim,)
            print(f"Stored mean initial values for generation: {self.mean_initial_values}")
    
    def generate(self, num_samples, initial_value=None, output_length=None,
                seed=None, linear_timestamps=False, timestamps=None):
        """
        Generate `num_samples` synthetic time series.
        
        Args:
            num_samples: Number of synthetic samples to generate
            initial_value: Initial values for feature channels. If None, uses mean from training
            output_length: Length of generated sequences. Defaults to self.seq_length
            seed: Random seed for reproducibility
            linear_timestamps: If True, use evenly spaced timestamps
            timestamps: Explicit timestamps to use if provided
            
        Returns:
            Synthetic time series of shape (num_samples, output_length, num_channels)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        L = output_length or self.seq_length
        
        with torch.no_grad():
            # --- Generate timestamps ---
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
            
            # --- Generate features ---
            Z = torch.randn(num_samples, L, self.latent_dim, device=self.device)
            H_fake = self.generator(Z, delta_t)
            H_fake_sup = self.supervisor(H_fake)
            X_fake = self.recovery(H_fake_sup)  # Shape: (num_samples, L, feature_dim)
            
            # --- Condition on initial values ---
            if initial_value is None:
                if self.mean_initial_values is not None:
                    initial_value = self.mean_initial_values
                else:
                    initial_value = np.full((self.feature_dim,), 0.5, dtype=np.float32)
            else:
                initial_value = np.asarray(initial_value, dtype=np.float32)
                if initial_value.shape == ():
                    initial_value = np.full((self.feature_dim,), float(initial_value), dtype=np.float32)
                elif initial_value.shape[0] != self.feature_dim:
                    raise ValueError(f"initial_value must be scalar or length equal to {self.feature_dim} channels")
            
            initial_value_tensor = torch.tensor(initial_value, dtype=torch.float32, device=self.device)
            generated_first_value = X_fake[:, 0, :]
            offset = initial_value_tensor.unsqueeze(0) - generated_first_value
            
            # Apply offset to all timesteps
            X_fake = X_fake + offset.unsqueeze(1)  # Broadcast across time dimension
            
            # Concatenate timestamps and features
            fake_series = torch.cat([timestamps, X_fake], dim=-1)
        
        return fake_series.cpu().numpy()