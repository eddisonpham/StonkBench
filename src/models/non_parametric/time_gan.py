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
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        h = self.fc(rnn_out)
        return h


class Recovery(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(_weights_init)

    def forward(self, h, apply_sigmoid=True):
        rnn_out, _ = self.rnn(h)
        x_tilde = self.fc(rnn_out)
        if apply_sigmoid:
            x_tilde = torch.sigmoid(x_tilde)
        return x_tilde


class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, z):
        rnn_out, _ = self.rnn(z)
        e = self.fc(rnn_out)
        return e  # No sigmoid in latent space


class Supervisor(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_weights_init)

    def forward(self, h):
        rnn_out, _ = self.rnn(h)
        h_supervise = self.fc(rnn_out)
        return h_supervise  # No sigmoid in latent space


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.apply(_weights_init)

    def forward(self, h):
        rnn_out, _ = self.rnn(h)
        y_hat = self.fc(rnn_out)
        y_hat = torch.sigmoid(y_hat)
        return y_hat


class TimeGAN(DeepLearningModel):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        gamma: float = 1.0,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(seed=seed)

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # For univariate time series
        self.input_dim = 1
        self.output_dim = 1

        # Networks
        self.encoder = Encoder(self.input_dim, hidden_dim, num_layers).to(self.device)
        self.recovery = Recovery(hidden_dim, self.output_dim, num_layers).to(self.device)
        self.generator = Generator(self.input_dim, hidden_dim, num_layers).to(self.device)
        self.supervisor = Supervisor(hidden_dim, num_layers).to(self.device)
        self.discriminator = Discriminator(hidden_dim, num_layers).to(self.device)

        # Losses
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Optimizers
        self.optimizer_autoencoder = None
        self.optimizer_supervisor = None
        self.optimizer_generator = None
        self.optimizer_discriminator = None

        # Data statistics
        self.data_min = 0.0
        self.data_max = 1.0

    def _init_optimizers(self):
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
        return (data - self.data_min) / (self.data_max - self.data_min + 1e-8)

    def _denormalize_data(self, data: torch.Tensor) -> torch.Tensor:
        return data * (self.data_max - self.data_min + 1e-8) + self.data_min

    def _train_autoencoder(self, data_loader, num_iterations: int):
        print("\n=== Stage 1: Training Autoencoder ===")
        self.encoder.train()
        self.recovery.train()

        for iteration in range(num_iterations):
            epoch_loss = 0.0
            num_batches = 0
            for batch in data_loader:
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)
                x = batch.to(self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)

                x_norm = self._normalize_data(x)
                h = self.encoder(x_norm)
                x_tilde = self.recovery(h)
                loss = self.mse_loss(x_tilde, x_norm)

                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (iteration + 1) % 100 == 0:
                print(f"Epoch {iteration+1}/{num_iterations}, Loss: {epoch_loss/num_batches:.6f}")

    def _train_supervisor(self, data_loader, num_iterations: int):
        print("\n=== Stage 2: Training Supervisor ===")
        self.supervisor.train()
        self.encoder.eval()

        for iteration in range(num_iterations):
            epoch_loss = 0.0
            num_batches = 0
            for batch in data_loader:
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)
                x = batch.to(self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)

                x_norm = self._normalize_data(x)
                with torch.no_grad():
                    h = self.encoder(x_norm)

                if h.size(1) < 2:
                    continue

                h_supervise = self.supervisor(h)
                loss = self.mse_loss(h_supervise[:, :-1, :], h[:, 1:, :])

                self.optimizer_supervisor.zero_grad()
                loss.backward()
                self.optimizer_supervisor.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (iteration + 1) % 100 == 0:
                print(f"Epoch {iteration+1}/{num_iterations}, Loss: {epoch_loss/num_batches:.6f}")

    def _train_adversarial(self, data_loader, num_iterations: int):
        print("\n=== Stage 3: Adversarial Training ===")
        self.encoder.eval()
        self.recovery.eval()

        for iteration in range(num_iterations):
            g_loss_total = 0.0
            d_loss_total = 0.0
            num_batches = 0

            for batch in data_loader:
                if batch.dim() == 1:
                    batch = batch.unsqueeze(0)
                x = batch.to(self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)

                batch_size = x.size(0)
                x_norm = self._normalize_data(x)

                # ------------------
                # Generator + Supervisor
                # ------------------
                self.generator.train()
                self.supervisor.train()
                z = torch.randn(batch_size, self.seq_len, self.input_dim).to(self.device)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                with torch.no_grad():
                    h = self.encoder(x_norm)
                    x_recon = self.recovery(h)

                # Generator adversarial loss
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)
                g_loss_u = self.bce_loss(y_fake, torch.ones_like(y_fake))
                g_loss_u_e = self.bce_loss(y_fake_e, torch.ones_like(y_fake_e))

                # Supervised loss
                if h_hat.size(1) > 1:
                    g_loss_s = self.mse_loss(h_hat[:, :-1, :], e_hat[:, 1:, :])
                else:
                    g_loss_s = 0.0

                # Moment matching
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
                # Discriminator
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

            if (iteration + 1) % 100 == 0:
                print(f"Epoch {iteration+1}/{num_iterations}, "
                      f"G Loss: {g_loss_total/num_batches:.6f}, "
                      f"D Loss: {d_loss_total/num_batches:.6f}")

    def fit(self, data_loader, autoencoder_iterations=1000, supervisor_iterations=1000, adversarial_iterations=1000):
        print("Starting TimeGAN Training...")
        self._init_optimizers()

        # Compute data min/max
        data_min_val = float('inf')
        data_max_val = float('-inf')
        for batch in data_loader:
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
            batch_min = batch.min().item()
            batch_max = batch.max().item()
            data_min_val = min(data_min_val, batch_min)
            data_max_val = max(data_max_val, batch_max)
        self.data_min = data_min_val
        self.data_max = data_max_val
        print(f"Data range: [{self.data_min:.4f}, {self.data_max:.4f}]")

        # Stage 1
        self._train_autoencoder(data_loader, autoencoder_iterations)
        # Stage 2
        self._train_supervisor(data_loader, supervisor_iterations)
        # Stage 3
        self._train_adversarial(data_loader, adversarial_iterations)
        print("Training complete!")

    def generate(self, num_samples: int, generation_length: int):
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
            return x_hat.squeeze(-1).cpu()
