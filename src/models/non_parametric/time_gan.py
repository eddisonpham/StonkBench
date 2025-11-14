import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from pathlib import Path
import sys
# Add parent directory to path to import base model
sys.path.append(str(Path(__file__).parent.parent))
from base.base_model import DeepLearningModel


def _weights_init(m):
    """Initialize network weights using Xavier/Orthogonal initialization."""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif 'Conv' in classname:
        init.normal_(m.weight, 0.0, 0.02)
    elif 'Norm' in classname:
        init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif 'GRU' in classname:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    """Encoder (Embedder) maps time series to latent space."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)
        h = self.fc(rnn_out)  # (batch_size, seq_len, hidden_dim)
        return h


class Recovery(nn.Module):
    """Recovery maps latent space back to time series."""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(_weights_init)

    def forward(self, h, apply_sigmoid=True):
        # h: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(h)
        x_tilde = self.fc(rnn_out)  # (batch_size, seq_len, output_dim)
        if apply_sigmoid:
            x_tilde = torch.sigmoid(x_tilde)
        return x_tilde


class Generator(nn.Module):
    """Generator generates synthetic latent representations."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, z):
        # z: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(z)
        e = self.fc(rnn_out)  # (batch_size, seq_len, hidden_dim)
        return e


class Supervisor(nn.Module):
    """Supervisor predicts next step in latent space."""
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, h):
        # h: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(h)
        h_supervise = self.fc(rnn_out)  # (batch_size, seq_len, hidden_dim)
        return h_supervise


class Discriminator(nn.Module):
    """Discriminator distinguishes real vs synthetic latent representations."""
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(_weights_init)

    def forward(self, h):
        # h: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(h)
        y_hat = self.fc(rnn_out)  # (batch_size, seq_len, 1)
        y_hat = torch.sigmoid(y_hat)
        return y_hat


class TimeGAN(DeepLearningModel):
    """
    TimeGAN model for generating synthetic time series.
    
    Input: DataLoader providing batches of shape (batch_size, seq_length)
    Output: Generated samples of shape (num_samples, generation_length)
    """
    
    def __init__(
        self,
        seq_len: int = None,
        hidden_dim: int = 24,
        num_layers: int = 3,
        gamma: float = 1.0,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.seq_len = seq_len  # Will be inferred from data if None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # For univariate time series
        self.input_dim = 1
        self.output_dim = 1

        # Networks (will be initialized in fit if seq_len is None)
        self.encoder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None

        # Losses
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Optimizers
        self.optimizer_autoencoder = None
        self.optimizer_supervisor = None
        self.optimizer_generator = None
        self.optimizer_discriminator = None

        # Data statistics for normalization
        self.data_min = 0.0
        self.data_max = 1.0

    def _init_networks(self):
        """Initialize networks if not already initialized."""
        if self.encoder is None:
            self.encoder = Encoder(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
            self.recovery = Recovery(self.hidden_dim, self.output_dim, self.num_layers).to(self.device)
            self.generator = Generator(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
            self.supervisor = Supervisor(self.hidden_dim, self.num_layers).to(self.device)
            self.discriminator = Discriminator(self.hidden_dim, self.num_layers).to(self.device)

    def _init_optimizers(self):
        """Initialize optimizers."""
        self.optimizer_autoencoder = optim.Adam(
            list(self.encoder.parameters()) + list(self.recovery.parameters()),
            lr=self.learning_rate
        )
        self.optimizer_supervisor = optim.Adam(
            self.supervisor.parameters(),
            lr=self.learning_rate
        )
        self.optimizer_generator = optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.learning_rate
        )
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate
        )

    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to [0, 1] range (min-max normalization)."""
        return (data - self.data_min) / (self.data_max - self.data_min + 1e-8)

    def _denormalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data from [0, 1] range back to original scale."""
        return data * (self.data_max - self.data_min + 1e-8) + self.data_min

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert batch from (batch_size, seq_len) to (batch_size, seq_len, 1)."""
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        x = batch.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        return x

    def _train_autoencoder(self, data_loader, num_iterations: int):
        """Stage 1: Train autoencoder (encoder + recovery)."""
        self.encoder.train()
        self.recovery.train()

        for iteration in range(num_iterations):
            epoch_loss = 0.0
            num_batches = 0
            for batch in data_loader:
                x = self._prepare_batch(batch)
                x_norm = self._normalize_data(x)
                
                h = self.encoder(x_norm)
                x_tilde = self.recovery(h)
                loss = self.mse_loss(x_tilde, x_norm)

                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (iteration + 1) % max(1, num_iterations // 10) == 0:
                print(f"  Autoencoder iteration {iteration+1}/{num_iterations}, Loss: {epoch_loss/num_batches:.6f}")

    def _train_supervisor(self, data_loader, num_iterations: int):
        """Stage 2: Train supervisor."""
        self.supervisor.train()
        self.encoder.eval()

        for iteration in range(num_iterations):
            epoch_loss = 0.0
            num_batches = 0
            for batch in data_loader:
                x = self._prepare_batch(batch)
                x_norm = self._normalize_data(x)
                
                with torch.no_grad():
                    h = self.encoder(x_norm)

                if h.size(1) < 2:
                    continue

                h_supervise = self.supervisor(h)
                # Supervised loss: predict next step
                loss = self.mse_loss(h_supervise[:, :-1, :], h[:, 1:, :])

                self.optimizer_supervisor.zero_grad()
                loss.backward()
                self.optimizer_supervisor.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (iteration + 1) % max(1, num_iterations // 10) == 0:
                print(f"  Supervisor iteration {iteration+1}/{num_iterations}, Loss: {epoch_loss/num_batches:.6f}")

    def _train_adversarial(self, data_loader, num_iterations: int):
        """Stage 3: Adversarial training (generator + discriminator)."""
        self.encoder.eval()
        self.recovery.eval()

        for iteration in range(num_iterations):
            g_loss_total = 0.0
            d_loss_total = 0.0
            num_batches = 0

            for batch in data_loader:
                x = self._prepare_batch(batch)
                batch_size = x.size(0)
                x_norm = self._normalize_data(x)

                # ------------------
                # Train Generator + Supervisor
                # ------------------
                self.generator.train()
                self.supervisor.train()
                
                # Generate synthetic data
                z = torch.randn(batch_size, self.seq_len, self.input_dim).to(self.device)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                with torch.no_grad():
                    h = self.encoder(x_norm)
                    x_recon = self.recovery(h)

                # Generator losses
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)
                g_loss_u = self.bce_loss(y_fake, torch.ones_like(y_fake))
                g_loss_u_e = self.bce_loss(y_fake_e, torch.ones_like(y_fake_e))

                # Supervised loss
                if h_hat.size(1) > 1:
                    g_loss_s = self.mse_loss(h_hat[:, :-1, :], e_hat[:, 1:, :])
                else:
                    g_loss_s = 0.0

                # Moment matching loss
                g_loss_v1 = torch.mean(torch.abs(x_hat.var(dim=0) - x_norm.var(dim=0)))
                g_loss_v2 = torch.mean(torch.abs(x_hat.mean(dim=0) - x_norm.mean(dim=0)))

                # Reconstruction loss
                g_loss_recon = self.mse_loss(x_recon, x_norm)

                g_loss = (g_loss_u + self.gamma * g_loss_u_e +
                          10 * g_loss_s +
                          5 * g_loss_v1 + 5 * g_loss_v2 +
                          10 * g_loss_recon)

                self.optimizer_generator.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), 5)
                self.optimizer_generator.step()
                g_loss_total += g_loss.item()

                # ------------------
                # Train Discriminator
                # ------------------
                self.discriminator.train()
                with torch.no_grad():
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    h_real = self.encoder(x_norm)

                y_real = self.discriminator(h_real)
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)

                d_loss_real = self.bce_loss(y_real, torch.ones_like(y_real))
                d_loss_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                d_loss_fake_e = self.bce_loss(y_fake_e, torch.zeros_like(y_fake_e))
                d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e

                self.optimizer_discriminator.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
                self.optimizer_discriminator.step()
                d_loss_total += d_loss.item()

                num_batches += 1

            if (iteration + 1) % max(1, num_iterations // 10) == 0:
                print(f"  Adversarial iteration {iteration+1}/{num_iterations}, "
                      f"G Loss: {g_loss_total/num_batches:.6f}, "
                      f"D Loss: {d_loss_total/num_batches:.6f}")

    def fit(self, data_loader, num_epochs: int = 10, *args, **kwargs):
        """
        Train TimeGAN model.

        Args:
            data_loader: DataLoader providing batches of shape (batch_size, seq_length)
            num_epochs: Number of training epochs (controls iterations per stage)
            supervisor_epochs: Number of supervisor training epochs (default: num_epochs)
            adversarial_epochs: Number of adversarial training epochs (default: num_epochs)
        """
        # Extract extra epoch options for supervisor and adversarial training
        supervisor_epochs = num_epochs // 2
        adversarial_epochs = num_epochs * 2

        # Infer seq_len from first batch if not set
        data_min_val = float('inf')
        data_max_val = float('-inf')
        data_loader_for_min_max = data_loader

        # First sweep for min/max normalization over all batches
        for batch in data_loader_for_min_max:
            batch_tensor = batch if isinstance(batch, torch.Tensor) else torch.tensor(batch)
            batch_min = batch_tensor.min().item()
            batch_max = batch_tensor.max().item()
            data_min_val = min(data_min_val, batch_min)
            data_max_val = max(data_max_val, batch_max)
        self.data_min = data_min_val
        self.data_max = data_max_val
        print(f"Data range (for normalization): [{self.data_min:.4f}, {self.data_max:.4f}]")

        # Infer seq_len from first batch if not set
        if self.seq_len is None:
            first_batch = next(iter(data_loader))
            self.seq_len = first_batch.shape[-1] if first_batch.dim() >= 1 else len(first_batch)
            print(f"Inferred sequence length: {self.seq_len}")

        # Initialize networks
        self._init_networks()
        self._init_optimizers()

        # === 3-stage TimeGAN training === #
        print("\n=== Stage 1: Training Autoencoder ===")
        self._train_autoencoder(data_loader, num_epochs)

        print("\n=== Stage 2: Training Supervisor ===")
        self._train_supervisor(data_loader, supervisor_epochs)

        print("\n=== Stage 3: Adversarial Training ===")
        self._train_adversarial(data_loader, adversarial_epochs)

        print("TimeGAN training complete!")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        """
        Generate synthetic time series samples.
        
        Args:
            num_samples: Number of samples to generate
            generation_length: Length of each generated sequence
            seed: Random seed for generation
        Returns:
            Generated samples of shape (num_samples, generation_length)
        """
        torch.manual_seed(seed)
        if self.generator is None:
            raise RuntimeError("Model must be trained before generating samples.")
        
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, generation_length, self.input_dim).to(self.device)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)
            x_hat = torch.clamp(x_hat, 0.0, 1.0)
            x_hat = self._denormalize_data(x_hat)
            return x_hat.squeeze(-1).cpu()  # (num_samples, generation_length)
