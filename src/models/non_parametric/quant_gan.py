import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.base.base_model import DeepLearningModel


class QuantGAN(DeepLearningModel):
    def __init__(self, seq_length, num_features, embedding_dim=64, hidden_dim=128, num_layers=2, kernel_size=3, padding=1, batch_size=32, learning_rate=2e-4):
        super().__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------------
        # Generator
        # -------------------------------
        self.generator = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_features, kernel_size=kernel_size, padding=padding)
        )

        # -------------------------------
        # Discriminator
        # -------------------------------
        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=kernel_size, padding=padding)
        )

        # Optimizers
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        # Losses
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss()

        self.to(self.device)

    def fit(self, data_loader: DataLoader, epochs=10):
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for real_data in data_loader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Transpose real_data from [batch, seq_len, features] to [batch, features, seq_len]
                real_data = real_data.transpose(1, 2)

                # -------------------------------
                # 1. Generator forward pass
                # -------------------------------
                noise = torch.randn(batch_size, self.embedding_dim, self.seq_length, device=self.device)
                generated_data = self.generator(noise)

                # -------------------------------
                # 2. Train Discriminator
                # -------------------------------
                self.opt_disc.zero_grad()
                
                real_out = self.discriminator(real_data)
                fake_out = self.discriminator(generated_data.detach())

                real_loss = self.adv_criterion(real_out, torch.ones_like(real_out))
                fake_loss = self.adv_criterion(fake_out, torch.zeros_like(fake_out))
                disc_loss = (real_loss + fake_loss) / 2

                disc_loss.backward()
                self.opt_disc.step()

                # -------------------------------
                # 3. Train Generator
                # -------------------------------
                self.opt_gen.zero_grad()
                
                # Recompute discriminator output for generator training (without detach)
                fake_out_gen = self.discriminator(generated_data)
                gen_loss = self.adv_criterion(fake_out_gen, torch.ones_like(fake_out_gen))
                
                gen_loss.backward()
                self.opt_gen.step()

                epoch_loss += disc_loss.item() + gen_loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(data_loader):.4f}")

    def generate(self, num_samples: int, seq_length: int = None):
        self.eval()
        seq_length = seq_length or self.seq_length
        noise = torch.randn(num_samples, self.embedding_dim, seq_length, device=self.device)
        with torch.no_grad():
            generated_data = self.generator(noise)
        # Transpose from [batch, features, seq_len] to [batch, seq_len, features]
        generated_data = generated_data.transpose(1, 2)
        return generated_data.cpu()
