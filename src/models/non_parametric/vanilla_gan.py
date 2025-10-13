import torch
import torch.nn as nn
import torch.optim as optim
from src.models.base.base_model import DeepLearningModel  # your base class


class TimeConditionedGenerator(nn.Module):
    def __init__(self, latent_dim, length, channels, hidden_dim=128):
        super().__init__()
        self.length = length
        self.channels = channels

        # We'll generate only the signal channels; timestamp is kept fixed
        self.signal_channels = channels - 1  # exclude timestamp

        self.model = nn.Sequential(
            nn.Linear((latent_dim + 1) * length, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, length * self.signal_channels),
        )

    def forward(self, z, timestamps):
        """
        z: (batch, latent_dim)
        timestamps: (batch, length, 1), unevenly spaced, normalized
        """
        batch_size = z.size(0)
        z_expanded = z.unsqueeze(1).repeat(1, self.length, 1)  # (batch, length, latent_dim)
        x = torch.cat([z_expanded, timestamps], dim=-1)  # (batch, length, latent_dim + 1)
        x = x.view(batch_size, -1)
        out_signals = self.model(x)  # (batch, length * signal_channels)
        out_signals = out_signals.view(batch_size, self.length, self.signal_channels)
        # Concatenate timestamp channel back
        out = torch.cat([timestamps, out_signals], dim=-1)
        return out


class TimeConditionedDiscriminator(nn.Module):
    def __init__(self, length, channels, hidden_dim=128):
        super().__init__()
        self.length = length
        self.channels = channels
        self.model = nn.Sequential(
            nn.Linear(channels * length, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: (batch, length, channels), including timestamp as 0-th channel
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.model(x_flat)


class VanillaGAN(DeepLearningModel):
    def __init__(self, length, num_channels, latent_dim=64, hidden_dim=128,
                 lr=0.0002, b1=0.5, b2=0.999):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.generator = TimeConditionedGenerator(latent_dim, length, num_channels, hidden_dim).to(self.device)
        self.discriminator = TimeConditionedDiscriminator(length, num_channels, hidden_dim).to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.bce_loss = nn.BCELoss()
        self.real_label = 1.0
        self.fake_label = 0.0

    def fit(self, data_loader, num_epochs=100):
        for epoch in range(num_epochs):
            for i, real_data in enumerate(data_loader):
                real_data = real_data.to(self.device)
                timestamps = real_data[:, :, 0:1]  # keep timestamp
                batch_size = real_data.size(0)

                # === Train Discriminator ===
                self.optimizer_d.zero_grad()

                # Real data
                label = torch.full((batch_size,), self.real_label, device=self.device)
                output_real = self.discriminator(real_data).view(-1)
                err_d_real = self.bce_loss(output_real, label)
                err_d_real.backward()

                # Fake data
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise, timestamps)
                label.fill_(self.fake_label)
                output_fake = self.discriminator(fake_data.detach()).view(-1)
                err_d_fake = self.bce_loss(output_fake, label)
                err_d_fake.backward()

                err_d = err_d_real + err_d_fake
                self.optimizer_d.step()

                # === Train Generator ===
                self.optimizer_g.zero_grad()
                label.fill_(self.real_label)
                output_fake_for_g = self.discriminator(fake_data).view(-1)
                err_g = self.bce_loss(output_fake_for_g, label)
                err_g.backward()
                self.optimizer_g.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(data_loader)}] - "
                          f"D Loss: {err_d.item():.4f}, G Loss: {err_g.item():.4f}")

    def generate(self, num_samples, timestamps):
        """
        timestamps: (num_samples, length, 1), already normalized
        """
        timestamps = timestamps.to(self.device)
        noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            generated = self.generator(noise, timestamps).cpu()
        return generated

