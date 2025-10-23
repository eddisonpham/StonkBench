import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.base.base_model import DeepLearningModel


class TimeGAN(DeepLearningModel):
    def __init__(self, seq_length, num_features, embedding_dim=64, hidden_dim=128, num_layers=2, batch_size=32, learning_rate=2e-4):
        super().__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------------------------------
        # Encoder (embedding network)
        # -------------------------------
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.enc_out = nn.Linear(hidden_dim, embedding_dim)

        # -------------------------------
        # Decoder (recovery network)
        # -------------------------------
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, num_features)

        # -------------------------------
        # Generator
        # -------------------------------
        self.generator = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.gen_out = nn.Linear(hidden_dim, embedding_dim)

        # -------------------------------
        # Discriminator
        # -------------------------------
        self.discriminator = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dis_out = nn.Linear(hidden_dim, 1)

        # Optimizers
        self.opt_enc_dec = optim.Adam(list(self.encoder.parameters()) + list(self.enc_out.parameters()) +
                                      list(self.decoder.parameters()) + list(self.dec_out.parameters()), lr=learning_rate)
        self.opt_gen = optim.Adam(list(self.generator.parameters()) + list(self.gen_out.parameters()), lr=learning_rate)
        self.opt_disc = optim.Adam(list(self.discriminator.parameters()) + list(self.dis_out.parameters()), lr=learning_rate)

        # Losses
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss()
        self.supervised_criterion = nn.MSELoss()  # For latent supervised loss

        self.to(self.device)

    def fit(self, data_loader: DataLoader, epochs=10):
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for real_data in data_loader:
                real_data = real_data.to(self.device)  # (batch, seq, features)
                batch_size = real_data.size(0)

                # -------------------------------
                # 1. Encoder -> latent_real
                # -------------------------------
                enc_out, _ = self.encoder(real_data)
                latent_real = self.enc_out(enc_out)  # (batch, seq, embedding_dim)

                # -------------------------------
                # 2. Decoder -> reconstruct input
                # -------------------------------
                dec_out, _ = self.decoder(latent_real)
                recon_data = self.dec_out(dec_out)
                recon_loss = self.recon_criterion(recon_data, real_data)

                # -------------------------------
                # 3. Supervised loss: generator should map noise to latent_real
                # -------------------------------
                noise = torch.randn(batch_size, self.seq_length, self.embedding_dim, device=self.device)
                gen_out, _ = self.generator(noise)
                latent_fake = self.gen_out(gen_out)
                supervised_loss = self.supervised_criterion(latent_fake, latent_real.detach())

                # -------------------------------
                # 4. Update encoder-decoder (reconstruction)
                # -------------------------------
                self.opt_enc_dec.zero_grad()
                recon_loss.backward(retain_graph=True)
                self.opt_enc_dec.step()

                # -------------------------------
                # 5. Update discriminator
                # -------------------------------
                disc_real_out, _ = self.discriminator(latent_real.detach())
                disc_real_out = self.dis_out(disc_real_out).mean(dim=1)  # sequence-level
                disc_fake_out, _ = self.discriminator(latent_fake.detach())
                disc_fake_out = self.dis_out(disc_fake_out).mean(dim=1)

                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                disc_loss = 0.5 * (self.adv_criterion(disc_real_out, real_labels) +
                                   self.adv_criterion(disc_fake_out, fake_labels))

                self.opt_disc.zero_grad()
                disc_loss.backward()
                self.opt_disc.step()

                # -------------------------------
                # 6. Update generator
                # -------------------------------
                gen_out, _ = self.generator(noise)
                latent_fake = self.gen_out(gen_out)
                disc_fake_out, _ = self.discriminator(latent_fake)
                disc_fake_out = self.dis_out(disc_fake_out).mean(dim=1)
                adv_loss = self.adv_criterion(disc_fake_out, real_labels)

                gen_loss = adv_loss + supervised_loss  # combine adversarial + supervised

                self.opt_gen.zero_grad()
                gen_loss.backward()
                self.opt_gen.step()

                epoch_loss += (recon_loss.item() + disc_loss.item() + gen_loss.item())

            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(data_loader):.4f}")

    def generate(self, num_samples: int, seq_length: int = None):
        self.eval()
        seq_length = seq_length or self.seq_length
        noise = torch.randn(num_samples, seq_length, self.embedding_dim, device=self.device)
        with torch.no_grad():
            gen_out, _ = self.generator(noise)
            latent_fake = self.gen_out(gen_out)
            dec_out, _ = self.decoder(latent_fake)
            generated_data = self.dec_out(dec_out)
        return generated_data.cpu()  # (num_samples, seq_length, num_features)
