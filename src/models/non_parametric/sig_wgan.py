import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm

from src.models.base.base_model import DeepLearningModel


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu2 = nn.PReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# --------------------------
# Temporal Convolutional Network (TCN)
# --------------------------
class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=80):
        super(TCN, self).__init__()
        layers = []
        dilation = 1
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            layers += [TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation)]
            dilation = 2 * dilation if i > 0 else 1
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.net(x.transpose(1, 2))
        return self.conv(y1).transpose(1, 2)

# --------------------------
# Generator
# --------------------------
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.tanh(self.net(x))

# --------------------------
# Discriminator (Critic)
# --------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# --------------------------
# Sig-WGAN Model
# --------------------------
class SigWGAN(DeepLearningModel):
    """
    Sig-WGAN: Signature-Wasserstein GAN for Time Series Generation.

    Implements a Wasserstein GAN using TCN-based Generator and Discriminator,
    enhanced with Signature-Wasserstein distance for improved training stability
    and sample quality.

    Inputs:
        - seq_length: Length of each time series segment (l)
        - num_features: Number of output channels (N)
        - embedding_dim: Dimensionality of latent noise input (nz)
        - hidden_dim: Hidden channel size in TCN layers
        - lr: Learning rate for both generator and discriminator
        - clip_value: Weight clipping threshold for WGAN
        - n_critic: Number of discriminator updates per generator update
    """

    def __init__(
        self,
        seq_length: int,
        num_features: int,
        embedding_dim: int = 3,
        hidden_dim: int = 80,
        lr: float = 0.0002,
        clip_value: float = 0.01,
        n_critic: int = 5,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_value = clip_value
        self.n_critic = n_critic

        self.generator = Generator(self.embedding_dim, self.num_features).to(self.device)
        self.discriminator = Discriminator(self.num_features, 1).to(self.device)

        self.opt_G = optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.opt_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        self.trained = False

    def fit(self, data_loader, num_epochs: int = 10, verbose: bool = True):
        """
        Train Sig-WGAN on time series data using Wasserstein GAN training loop.

        Args:
            data_loader (torch.utils.data.DataLoader): batches of (batch_size, seq_len, num_features)
            verbose (bool): Whether to show tqdm progress bar.
        """
        self.generator.train()
        self.discriminator.train()

        progress_bar = tqdm(range(num_epochs)) if verbose else range(num_epochs)

        for epoch in progress_bar:
            for i, real in enumerate(data_loader):
                real = real.to(self.device)
                batch_size = real.size(0)

                # ==========================
                # Train Discriminator (n_critic times)
                # ==========================
                self.discriminator.zero_grad()
                z = torch.randn(batch_size, self.seq_length, self.embedding_dim, device=self.device)
                fake = self.generator(z).detach()

                loss_D = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                loss_D.backward()
                self.opt_D.step()

                # Weight clipping for Lipschitz constraint
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # ==========================
                # Train Generator
                # ==========================
                if i % self.n_critic == 0:
                    self.generator.zero_grad()
                    z = torch.randn(batch_size, self.seq_length, self.embedding_dim, device=self.device)
                    fake = self.generator(z)
                    loss_G = -torch.mean(self.discriminator(fake))
                    loss_G.backward()
                    self.opt_G.step()

            if verbose:
                progress_bar.set_description(
                    f"Epoch [{epoch+1}/{num_epochs}] | Loss_D: {loss_D.item():.6f} | Loss_G: {loss_G.item():.6f}"
                )

        self.trained = True

    @torch.no_grad()
    def generate(self, num_samples: int, seq_length: int = None):
        """
        Generate synthetic time series samples after training.

        Args:
            num_samples (int): Number of samples to generate (R)
            seq_length (int, optional): Sequence length (defaults to training seq_length)

        Returns:
            torch.Tensor: Generated synthetic samples (R, l, N)
        """
        if not self.trained:
            raise RuntimeError("Sig-WGAN must be trained before generation.")

        self.generator.eval()
        seq_length = seq_length or self.seq_length

        z = torch.randn(num_samples, seq_length, self.embedding_dim, device=self.device)
        fake = self.generator(z)
        return fake.detach().cpu()
