import torch
import torch.nn as nn
import torch.optim as optim
from models.base_model import DeepLearningModel

# Reusing the Generator from VanillaGAN as it's typically the same structure
# or a modified version depending on the specific WGAN variant (e.g., WGAN-GP)
class Generator(nn.Module):
    def __init__(self, latent_dim, output_length, output_channels, hidden_dim=128):
        super().__init__()
        self.output_length = output_length
        self.output_channels = output_channels
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_length * output_channels),
            nn.Sigmoid(), # Ensure output is in [0, 1] for normalized time series
        )

    def forward(self, z):
        return self.model(z).view(-1, self.output_length, self.output_channels)

# WGAN Critic (Discriminator without sigmoid output)
class Critic(nn.Module):
    def __init__(self, input_length, input_channels, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length * input_channels, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1), # No sigmoid here
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class WassersteinGAN(DeepLearningModel):
    def __init__(self, length, num_channels, latent_dim=64, hidden_dim=128, lr=0.00005, n_critic=5, clip_value=0.01):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_critic = n_critic  # Number of critic updates per generator update
        self.clip_value = clip_value # Weight clipping value

        self.generator = Generator(latent_dim, length, num_channels, hidden_dim).to(self.device)
        self.critic = Critic(length, num_channels, hidden_dim).to(self.device)

        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_c = optim.RMSprop(self.critic.parameters(), lr=lr)
        

    def fit(self, data_loader, num_epochs=100):
        for epoch in range(num_epochs):
            for i, real_data in enumerate(data_loader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)

                # Train Critic
                for _ in range(self.n_critic):
                    self.optimizer_c.zero_grad()

                    # Real data
                    c_real = self.critic(real_data).view(-1)

                    # Fake data
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_data = self.generator(noise).detach() # Detach to prevent generator training
                    c_fake = self.critic(fake_data).view(-1)

                    # Wasserstein loss for critic
                    err_c = -torch.mean(c_real) + torch.mean(c_fake)
                    err_c.backward()
                    self.optimizer_c.step()

                    # Weight clipping
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                # Train Generator
                self.optimizer_g.zero_grad()
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise)
                g_fake = self.critic(fake_data).view(-1)

                # Wasserstein loss for generator
                err_g = -torch.mean(g_fake)
                err_g.backward()
                self.optimizer_g.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(data_loader)}] - C Loss: {err_c.item():.4f}, G Loss: {err_g.item():.4f}")

    def generate(self, num_samples):
        noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            generated_data = self.generator(noise).cpu()
        return generated_data
