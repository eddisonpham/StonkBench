import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from src.models.base.base_model import DeepLearningModel

class TimeGAN(DeepLearningModel):
    """
    TimeGAN implementation for time series generation.
    Based on "Time-series Generative Adversarial Networks" by Yoon et al.
    """
    
    def __init__(self, l, N, latent_dim=64, hidden_dim=128, lr=0.0002):
        super().__init__()
        
        self.l = l  # sequence length
        self.N = N  # number of features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Networks
        self.embedder = self._build_embedder()
        self.recovery = self._build_recovery()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Move to device
        self.embedder.to(self.device)
        self.recovery.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers
        self.opt_embedder = optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()), 
            lr=lr
        )
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)
        
    def _build_embedder(self):
        """Build the embedder network (X -> H)"""
        return nn.Sequential(
            nn.Linear(self.N, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_recovery(self):
        """Build the recovery network (H -> X)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.N),
            nn.Sigmoid()
        )
    
    def _build_generator(self):
        """Build the generator network (Z -> H)"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_discriminator(self):
        """Build the discriminator network (H -> probability)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def fit(self, data_loader, epochs=100):
        """
        Train the TimeGAN model
        
        Args:
            data_loader: DataLoader containing training data
            epochs: Number of training epochs
        """
        print(f"Training TimeGAN for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_e_loss = 0
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                else:
                    X = batch
                    
                X = X.to(self.device).float()
                batch_size = X.size(0)
                
                # Phase 1: Autoencoder Training (Embedder + Recovery)
                self.opt_embedder.zero_grad()
                
                H = self.embedder(X.view(-1, self.N)).view(batch_size, self.l, self.hidden_dim)
                X_tilde = self.recovery(H.view(-1, self.hidden_dim)).view(batch_size, self.l, self.N)
                
                e_loss = nn.MSELoss()(X_tilde, X)
                e_loss.backward()
                self.opt_embedder.step()
                
                # Phase 2: Supervised Loss (Generator)
                self.opt_generator.zero_grad()
                
                Z = torch.randn(batch_size, self.l, self.latent_dim, device=self.device)
                H_hat = self.generator(Z.view(-1, self.latent_dim)).view(batch_size, self.l, self.hidden_dim)
                X_hat = self.recovery(H_hat.view(-1, self.hidden_dim)).view(batch_size, self.l, self.N)
                
                # Supervised loss
                H_supervise = self.embedder(X_hat.view(-1, self.N)).view(batch_size, self.l, self.hidden_dim)
                g_loss_s = nn.MSELoss()(H_hat, H_supervise)
                
                g_loss_s.backward()
                self.opt_generator.step()
                
                # Phase 3: Discriminator Training
                self.opt_discriminator.zero_grad()
                
                # Real embeddings
                H_real = self.embedder(X.view(-1, self.N)).view(batch_size, self.l, self.hidden_dim)
                d_real = self.discriminator(H_real.view(-1, self.hidden_dim))
                
                # Fake embeddings
                Z = torch.randn(batch_size, self.l, self.latent_dim, device=self.device)
                H_fake = self.generator(Z.view(-1, self.latent_dim)).view(batch_size, self.l, self.hidden_dim)
                d_fake = self.discriminator(H_fake.detach().view(-1, self.hidden_dim))
                
                # Discriminator loss
                d_loss_real = nn.BCELoss()(d_real, torch.ones_like(d_real))
                d_loss_fake = nn.BCELoss()(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.opt_discriminator.step()
                
                # Phase 4: Generator Adversarial Training
                self.opt_generator.zero_grad()
                
                Z = torch.randn(batch_size, self.l, self.latent_dim, device=self.device)
                H_fake = self.generator(Z.view(-1, self.latent_dim)).view(batch_size, self.l, self.hidden_dim)
                d_fake = self.discriminator(H_fake.view(-1, self.hidden_dim))
                
                g_loss_adv = nn.BCELoss()(d_fake, torch.ones_like(d_fake))
                g_loss = g_loss_adv + g_loss_s
                
                g_loss.backward()
                self.opt_generator.step()
                
                # Track losses
                epoch_e_loss += e_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: E_Loss: {epoch_e_loss/num_batches:.4f}, "
                      f"G_Loss: {epoch_g_loss/num_batches:.4f}, D_Loss: {epoch_d_loss/num_batches:.4f}")
    
    def generate(self, num_samples, **kwargs):
        """
        Generate synthetic time series samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples as torch.Tensor of shape (num_samples, l, N)
        """
        self.eval()
        
        with torch.no_grad():
            # Generate random noise
            Z = torch.randn(num_samples, self.l, self.latent_dim, device=self.device)
            
            # Generate embeddings
            H_fake = self.generator(Z.view(-1, self.latent_dim)).view(num_samples, self.l, self.hidden_dim)
            
            # Recover to data space
            X_fake = self.recovery(H_fake.view(-1, self.hidden_dim)).view(num_samples, self.l, self.N)
        
        self.train()
        return X_fake
    
    def save_model(self, path):
        """Save the model state"""
        import os
        from src.utils.path_utils import make_sure_path_exist
        
        make_sure_path_exist(path)
        
        state_dict = {
            'embedder': self.embedder.state_dict(),
            'recovery': self.recovery.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'l': self.l,
            'N': self.N,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr
        }
        
        torch.save(state_dict, os.path.join(path, 'TimeGAN_model.pth'))
        print(f"TimeGAN model saved to {os.path.join(path, 'TimeGAN_model.pth')}")
    
    def load_model(self, path):
        """Load the model state"""
        import os
        
        state_dict = torch.load(os.path.join(path, 'TimeGAN_model.pth'), map_location=self.device)
        
        self.embedder.load_state_dict(state_dict['embedder'])
        self.recovery.load_state_dict(state_dict['recovery'])
        self.generator.load_state_dict(state_dict['generator'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        
        print(f"TimeGAN model loaded from {os.path.join(path, 'TimeGAN_model.pth')}")