from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base.base_model import DeepLearningModel


class Sampling(nn.Module):
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        batch_size = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch_size, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class TrendLayer(nn.Module):
    def __init__(self, latent_dim: int, feat_dim: int, trend_poly: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(latent_dim, feat_dim * trend_poly)
        self.trend_dense2 = nn.Linear(feat_dim * trend_poly, feat_dim * trend_poly)
        
    def forward(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        trend_params = torch.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)
        lin_space = torch.linspace(0, 1.0, seq_length, device=z.device)
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)
        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

class SeasonalLayer(nn.Module):
    def __init__(self, latent_dim: int, feat_dim: int, custom_seas: List[Tuple[int, int]]):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas
        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        
    def _get_season_indexes_over_seq(self, num_seasons: int, len_per_season: int, device: torch.device, seq_length: int) -> torch.Tensor:
        season_indexes = torch.arange(num_seasons, device=device).unsqueeze(1).expand(-1, len_per_season)
        season_indexes = season_indexes.reshape(-1)
        num_repeats = (seq_length // (num_seasons * len_per_season)) + 1
        season_indexes = season_indexes.repeat(num_repeats)[:seq_length]
        return season_indexes
    
    def forward(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        batch_size = z.size(0)
        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)
            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season, z.device, seq_length
            )
            season_indexes = season_indexes_over_time.unsqueeze(0).unsqueeze(0).expand(
                batch_size, self.feat_dim, -1
            )
            season_vals = torch.gather(
                season_params, dim=2, index=season_indexes
            )
            all_seas_vals.append(season_vals)
        all_seas_vals = torch.stack(all_seas_vals, dim=-1)
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)
        all_seas_vals = all_seas_vals.transpose(1, 2)
        return all_seas_vals

class TimeVAE(DeepLearningModel):
    def __init__(
        self,
        length: int,
        num_channels: int,
        latent_dim: int = 10,
        hidden_layer_sizes: Optional[List[int]] = None,
        trend_poly: int = 0,
        custom_seas: Optional[List[Tuple[int, int]]] = None,
        use_residual_conn: bool = True,
        reconstruction_wt: float = 3.0,
        lr: float = 1e-3,
    ):
        super().__init__(length=length, num_channels=num_channels)
        
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [100, 200, 400]
            
        self.latent_dim = int(latent_dim)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = int(trend_poly)
        self.custom_seas = custom_seas if custom_seas is not None else []
        self.use_residual_conn = bool(use_residual_conn)
        self.reconstruction_wt = float(reconstruction_wt)
        
        if not use_residual_conn and trend_poly == 0 and len(self.custom_seas) == 0:
            raise ValueError(
                "Error: No decoder model to use. "
                "You must use one or more of: "
                "trend, custom seasonality(ies), and/or residual connection."
            )
            
        self.sampling = Sampling()
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def _build_encoder(self) -> nn.Module:
        layers = []
        in_channels = self.num_channels
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            in_channels = num_filters
        self.conv_encoder = nn.Sequential(*layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_channels, self.length)
            dummy_output = self.conv_encoder(dummy_input)
            self.encoder_last_dense_dim = dummy_output.numel()
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, self.latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, self.latent_dim)
        class Encoder(nn.Module):
            def __init__(self, conv_encoder, z_mean, z_log_var, sampling):
                super().__init__()
                self.conv_encoder = conv_encoder
                self.z_mean = z_mean
                self.z_log_var = z_log_var
                self.sampling = sampling
                
            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.conv_encoder(x)
                x = x.reshape(x.size(0), -1)
                z_mean = self.z_mean(x)
                z_log_var = self.z_log_var(x)
                z = self.sampling(z_mean, z_log_var)
                return z_mean, z_log_var, z
        return Encoder(self.conv_encoder, self.z_mean, self.z_log_var, self.sampling)
    
    def _build_decoder(self) -> nn.Module:
        class ResidualModel(nn.Module):
            def __init__(self, latent_dim, encoder_last_dense_dim, hidden_layer_sizes, 
                         feat_dim):
                super().__init__()
                self.hidden_layer_sizes = hidden_layer_sizes
                self.feat_dim = feat_dim
                self.encoder_last_dense_dim = encoder_last_dense_dim
                self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
                spatial_dim = encoder_last_dense_dim // hidden_layer_sizes[-1]
                self.spatial_dim = spatial_dim
                layers = []
                in_channels = hidden_layer_sizes[-1]
                for num_filters in reversed(hidden_layer_sizes[:-1]):
                    layers.append(
                        nn.ConvTranspose1d(
                            in_channels=in_channels,
                            out_channels=num_filters,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                    )
                    layers.append(nn.ReLU())
                    in_channels = num_filters
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=in_channels,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
                self.deconv_layers = nn.Sequential(*layers)
            def forward(self, z, seq_length: int):
                x = torch.relu(self.dense(z))
                batch_size = x.size(0)
                x = x.view(batch_size, self.hidden_layer_sizes[-1], self.spatial_dim)
                x = self.deconv_layers(x)
                if x.size(2) != seq_length:
                    x = nn.functional.interpolate(x, size=seq_length, mode='linear', align_corners=False)
                residuals = x.transpose(1, 2)
                return residuals
        class Decoder(nn.Module):
            def __init__(self, latent_dim, feat_dim,
                         encoder_last_dense_dim, hidden_layer_sizes, 
                         trend_poly, custom_seas, use_residual_conn):
                super().__init__()
                self.feat_dim = feat_dim
                self.trend_layer = TrendLayer(latent_dim, feat_dim, trend_poly) if trend_poly > 0 else None
                self.seasonal_layer = SeasonalLayer(latent_dim, feat_dim, custom_seas) if custom_seas else None
                self.residual_model = ResidualModel(latent_dim, encoder_last_dense_dim,
                                                    hidden_layer_sizes, feat_dim) if use_residual_conn else None
            def forward(self, z, seq_length: int):
                outputs = torch.zeros(z.size(0), seq_length, self.feat_dim, device=z.device)
                if self.trend_layer is not None:
                    trend_vals = self.trend_layer(z, seq_length)
                    outputs = outputs + trend_vals
                if self.seasonal_layer is not None:
                    seas_vals = self.seasonal_layer(z, seq_length)
                    outputs = outputs + seas_vals
                if self.residual_model is not None:
                    residuals = self.residual_model(z, seq_length)
                    outputs = outputs + residuals
                return outputs
        return Decoder(
            self.latent_dim, self.num_channels,
            self.encoder_last_dense_dim, self.hidden_layer_sizes,
            self.trend_poly, self.custom_seas, self.use_residual_conn
        )
    
    def _reconstruction_loss(self, x: torch.Tensor, x_recons: torch.Tensor) -> torch.Tensor:
        err = (x - x_recons) ** 2
        reconst_loss = err.sum()
        x_recons_mean = x_recons.mean(dim=1)
        mean_reg_loss = (x_recons_mean ** 2).sum()
        total_reconst_loss = reconst_loss + mean_reg_loss
        return total_reconst_loss
    
    def _kl_loss(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        kl_loss = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        kl_loss = kl_loss.sum()
        return kl_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        x_recons = self.decoder(z, self.length)
        return x_recons, z_mean, z_log_var
    
    def fit(
        self,
        data_loader,
        num_epochs: int = 100,
        verbose: bool = True,
    ):
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_reconst_loss = 0.0
            total_kl_loss = 0.0
            num_batches = 0
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                x = batch.to(self.device)
                x_recons, z_mean, z_log_var = self.forward(x)
                reconstruction_loss = self._reconstruction_loss(x, x_recons)
                kl_loss = self._kl_loss(z_mean, z_log_var)
                total_loss_batch = self.reconstruction_wt * reconstruction_loss + kl_loss
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                total_loss += total_loss_batch.item()
                total_reconst_loss += reconstruction_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
            if verbose:
                avg_loss = total_loss / (num_batches * x.size(0)) if num_batches > 0 else 0.0
                avg_recon_loss = total_reconst_loss / (num_batches * x.size(0)) if num_batches > 0 else 0.0
                avg_kl_loss = total_kl_loss / (num_batches * x.size(0)) if num_batches > 0 else 0.0
                print(f"TimeVAE Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {avg_loss:.4f} | "
                      f"Recon Loss: {avg_recon_loss:.4f} | "
                      f"KL Loss: {avg_kl_loss:.4f}")
        self.eval()
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        seq_length: Optional[int] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        if seq_length is None:
            seq_length = self.length
        torch.manual_seed(seed)
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        self.eval()
        samples = self.decoder(z, seq_length)
        return samples.detach().cpu()

