"""
Takahashi Diffusion Model for Time Series Generation

This implementation follows the specifications in takahashi.md:
- Preprocessing: mirror expansion to power of 2, power transform, normalization, winsorization
- Wavelet transform with coefficient-to-image mapping
- Uses HuggingFace's DDPM (UNet2DModel + DDPMScheduler) as the diffusion base

For univariate time series (log returns), creates grayscale images from wavelet coefficients.
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pywt
from diffusers import UNet2DModel, DDPMScheduler

from src.models.base.base_model import DeepLearningModel


def mirror_expand_to_power_of_2(x: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Expand time series to nearest power of 2 using mirror reflection.
    """
    x = np.ravel(x)
    L = x.shape[0]
    n = math.ceil(math.log2(L))
    target_length = 2 ** n
    if L == target_length:
        return x, 0, L
    total_padding = target_length - L
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad
    left_reflection = np.flip(x[:left_pad]) if left_pad > 0 else np.array([])
    right_reflection = np.flip(x[-right_pad:]) if right_pad > 0 else np.array([])
    expanded = np.concatenate([left_reflection, x, right_reflection])
    original_start = left_pad
    original_end = left_pad + L
    return expanded, original_start, original_end


def power_transform_and_normalize(x: np.ndarray, power: float = 1.0) -> Tuple[np.ndarray, float, float]:
    """
    Apply power transformation and normalization.
    """
    mean_val = np.mean(x)
    std_val = np.std(x)
    if std_val < 1e-8:
        return (x - mean_val), mean_val, std_val
    centered = x - mean_val
    if abs(power - 1.0) < 1e-6:
        transformed = centered / std_val
    else:
        sign = np.sign(centered)
        abs_val = np.abs(centered)
        transformed = sign * (abs_val ** (1.0 / power)) / std_val
    return transformed, mean_val, std_val


def winsorize(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Winsorize outliers by clipping values beyond threshold.
    """
    return np.clip(x, -threshold, threshold)


def dwt_to_image_coefficients(x: np.ndarray, wavelet: str = 'haar') -> Tuple[np.ndarray, dict]:
    """
    Perform DWT and organize coefficients into image structure.
    
    For input length 2^n, DWT produces:
    - 0th-order: 1 coefficient
    - 1st-order: 1 coefficient  
    - 2nd-order: 2 coefficients
    - 3rd-order: 4 coefficients
    - ... up to (n-1)th order
    """
    L = x.shape[0]
    n = int(math.log2(L))
    coeffs = pywt.wavedec(x, wavelet, mode='zero', level=n - 1)
    image_rows = []
    row_length = L
    for coeff in coeffs:
        num_coeffs = len(coeff)
        segment_length = max(1, row_length // num_coeffs)
        row = np.repeat(coeff, segment_length)
        if row.shape[0] < row_length:
            row = np.pad(row, (0, row_length - row.shape[0]), mode='edge')
        elif row.shape[0] > row_length:
            row = row[:row_length]
        image_rows.append(row)
    image = np.stack(image_rows, axis=0)
    coeffs_info = {
        'original_length': L,
        'n_levels': len(coeffs),
        'coeff_lengths': [len(c) for c in coeffs],
        'wavelet': wavelet,
        'n': n
    }
    return image, coeffs_info


def image_coefficients_to_dwt(image: np.ndarray, coeffs_info: dict, wavelet: str = 'haar') -> np.ndarray:
    """
    Reconstruct time series from image coefficients using inverse DWT.
    
    Args:
        image: Image array of shape (n_levels, L)
        coeffs_info: Dictionary with coefficient information
        wavelet: Wavelet type
        
    Returns:
        reconstructed: Reconstructed time series
    """
    n_levels = coeffs_info['n_levels']
    coeff_lengths = coeffs_info['coeff_lengths']
    coeffs = []
    for i in range(n_levels):
        row = image[i]
        num_coeffs = coeff_lengths[i]
        segment_length = max(1, len(row) // num_coeffs)
        coeff = row[::segment_length][:num_coeffs]
        coeffs.append(coeff)
    reconstructed = pywt.waverec(coeffs, wavelet, mode='zero')
    return reconstructed

class TakahashiDiffusion(DeepLearningModel):
    """
    Takahashi Diffusion model for generating synthetic time series.
    
    Follows specifications in takahashi.md:
    - Preprocessing: mirror expansion, power transform, normalization, winsorization
    - Wavelet transform with coefficient-to-image mapping
    - Uses HuggingFace's DDPM for diffusion
    
    Input: DataLoader providing batches of shape (batch_size, seq_length)
    Output: Generated samples of shape (num_samples, generation_length)
    """
    
    def __init__(
        self,
        length: int = None,
        num_channels: int = 1,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        wavelet: str = 'haar',
        power_transform: float = 1.0,
        winsorize_threshold: float = 3.0,
        lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.length = length
        self.num_channels = int(num_channels)
        self.num_steps = int(num_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.wavelet = str(wavelet)
        self.power_transform = float(power_transform)
        self.winsorize_threshold = float(winsorize_threshold)
        self.lr = float(lr)
        self.device = torch.device(device)
        self.unet = None
        self.scheduler = None
        self.optimizer = None
        self.data_mean = None
        self.data_std = None
        self.expansion_info = None
        self.coeffs_info_cache = None
        
        # Store best parameters for model selection
        self.best_state_dict = None
        self.best_val_loss = float('inf')
        self._best_model_loaded = False

    def _init_model(self, image_height: int, image_width: int):
        """Initialize UNet2DModel and DDPMScheduler."""
        if self.unet is None:
            self.unet = UNet2DModel(
                sample_size=(image_height, image_width),
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(64, 128, 256),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    'AttnDownBlock2D',
                ),
                up_block_types=(
                    'AttnUpBlock2D',
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            ).to(self.device)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.num_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule="linear",
                prediction_type="epsilon",
            )
            self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr)

            # Report total number of trainable parameters for this Takahashi Diffusion UNet
            num_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            print(f"TakahashiDiffusion UNet trainable parameters: {num_params:,}")

    def _preprocess_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess batch according to takahashi.md specifications.
        
        Returns:
            images: Preprocessed images ready for DDPM (batch_size, 1, H, W)
            preprocess_info: Dictionary with preprocessing parameters for inverse
        """
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        elif batch.dim() > 2:
            batch = batch.view(batch.shape[0], -1)
        batch_np = batch.detach().cpu().numpy()
        batch_size = batch_np.shape[0]
        processed_images = []
        preprocess_infos = []
        for i in range(batch_size):
            x = batch_np[i]
            expanded, orig_start, orig_end = mirror_expand_to_power_of_2(x)
            transformed, mean_val, std_val = power_transform_and_normalize(
                expanded, self.power_transform
            )
            winsorized = winsorize(transformed, self.winsorize_threshold)
            image, coeffs_info = dwt_to_image_coefficients(winsorized, self.wavelet)
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            processed_images.append(image_tensor)
            preprocess_infos.append({
                'original_start': orig_start,
                'original_end': orig_end,
                'mean': mean_val,
                'std': std_val,
                'coeffs_info': coeffs_info
            })
        images = torch.stack(processed_images, dim=0).to(self.device)
        _, _, img_h, img_w = images.shape
        target_h = math.ceil(img_h / 8) * 8
        target_w = math.ceil(img_w / 8) * 8
        if (img_h != target_h) or (img_w != target_w):
            images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if self.coeffs_info_cache is None:
            self.coeffs_info_cache = preprocess_infos[0]['coeffs_info']
            self.expansion_info = (preprocess_infos[0]['original_start'], preprocess_infos[0]['original_end'])
            self.coeffs_info_cache['image_shape'] = (target_h, target_w)
            if self.data_mean is None:
                self.data_mean = preprocess_infos[0]['mean']
                self.data_std = preprocess_infos[0]['std']
        return images, preprocess_infos

    def _postprocess_images(self, images: torch.Tensor, preprocess_info: dict) -> torch.Tensor:
        """
        Reverse preprocessing pipeline to recover time series.
        
        Args:
            images: Generated images (batch_size, 1, H, W)
            preprocess_info: Preprocessing information for inverse transform
            
        Returns:
            time_series: Recovered time series (batch_size, original_length)
        """
        batch_size = images.shape[0]
        recovered_series = []
        for i in range(batch_size):
            image = images[i, 0].detach().cpu().numpy()
            info = preprocess_info[i] if isinstance(preprocess_info, list) else preprocess_info
            expanded = image_coefficients_to_dwt(image, info['coeffs_info'], self.wavelet)
            denormalized = expanded * info['std'] + info['mean']
            if abs(self.power_transform - 1.0) > 1e-6:
                sign = np.sign(denormalized)
                abs_val = np.abs(denormalized)
                denormalized = sign * (abs_val ** self.power_transform)
            orig_start = info['original_start']
            orig_end = info['original_end']
            original = denormalized[orig_start:orig_end]
            recovered_series.append(torch.from_numpy(original).float())
        return torch.stack(recovered_series, dim=0)

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert batch from (batch_size, seq_len) to (batch_size, seq_len, 1)."""
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        x = batch.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return x

    def fit(self, data_loader, num_epochs: int = 100, valid_loader=None, *args, **kwargs):
        """
        Train Takahashi Diffusion model.
        
        Args:
            data_loader: DataLoader providing batches of shape (batch_size, seq_length)
            num_epochs: Number of training epochs
            valid_loader: Optional DataLoader for validation set
        """
        try:
            first_batch, _ = next(iter(data_loader))
        except StopIteration:
            raise ValueError("The provided data_loader is empty.")
        if self.length is None:
            self.length = first_batch.shape[-1] if first_batch.dim() >= 1 else len(first_batch)
            print(f"Inferred sequence length: {self.length}")
        images, _ = self._preprocess_batch(first_batch)
        _, _, img_h, img_w = images.shape
        self._init_model(img_h, img_w)

        print(f"Image dimensions: {img_h} x {img_w}")
        
        # Initialize best_state_dict with current model state
        self.best_state_dict = {k: v.cpu().clone() for k, v in self.unet.state_dict().items()}
        self.best_val_loss = float('inf')
        self._best_model_loaded = False
        
        self.unet.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0
            for batch, _ in data_loader:
                images, _ = self._preprocess_batch(batch)
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=self.device
                ).long()
                noisy_images = self.scheduler.add_noise(images, noise, timesteps)
                noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            avg_loss = total_loss / batch_count
            
            # Compute validation loss if validation set is provided
            if valid_loader is not None:
                self.unet.eval()
                val_total_loss = 0.0
                val_batch_count = 0
                with torch.no_grad():
                    for batch, _ in valid_loader:
                        images, _ = self._preprocess_batch(batch)
                        noise = torch.randn_like(images)
                        timesteps = torch.randint(
                            0, self.scheduler.config.num_train_timesteps,
                            (images.shape[0],), device=self.device
                        ).long()
                        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
                        noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
                        val_loss = F.mse_loss(noise_pred, noise)
                        val_total_loss += val_loss.item()
                        val_batch_count += 1
                avg_val_loss = val_total_loss / val_batch_count
                
                # Save best model based on validation loss
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.unet.state_dict().items()}
                
                self.unet.train()
                print(f"TakahashiDiffusion epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                # Fall back to training loss if no validation set
                if avg_loss < self.best_val_loss:
                    self.best_val_loss = avg_loss
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.unet.state_dict().items()}
                print(f"TakahashiDiffusion epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Restore best model at the end
        if self.best_state_dict is not None:
            self.unet.load_state_dict(self.best_state_dict)
            self._best_model_loaded = True
            print(f"Best model restored with validation loss {self.best_val_loss:.6f}")
        
        self.unet.eval()

    @torch.no_grad()
    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        """
        Generate synthetic time series samples using Takahashi Diffusion.
        
        Uses cached preprocessing info from training to avoid zero-denormalization.
        
        Args:
            num_samples: Number of samples to generate
            generation_length: Length of each generated sequence
            seed: Random seed for generation
        
        Returns:
            Generated samples of shape (num_samples, generation_length)
        """
        if self.unet is None:
            raise RuntimeError("Model must be trained before generating samples.")
        if self.coeffs_info_cache is None:
            raise RuntimeError("Preprocessing info not available. Train the model first.")

        # Ensure best model is loaded
        if not self._best_model_loaded and self.best_state_dict is not None:
            self.unet.load_state_dict(self.best_state_dict)
            self._best_model_loaded = True

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.unet.eval()

        # Use cached preprocessing info
        dummy_info = {
            'coeffs_info': self.coeffs_info_cache,
            'original_start': self.expansion_info[0],
            'original_end': self.expansion_info[1],
            'mean': self.data_mean,
            'std': self.data_std
        }

        # Determine image size from cached coeffs info
        target_h, target_w = self.coeffs_info_cache['image_shape']
        generated_images = []

        for i in range(num_samples):
            torch.manual_seed(seed + i)
            # Start from pure noise
            image = torch.randn((1, 1, target_h, target_w), device=self.device)
            self.scheduler.set_timesteps(self.num_steps)
            for t in self.scheduler.timesteps:
                noise_pred = self.unet(image, t, return_dict=False)[0]
                image = self.scheduler.step(noise_pred, t, image, return_dict=False)[0]
            generated_images.append(image)

        images_batch = torch.cat(generated_images, dim=0)
        preprocess_infos = [dummy_info] * num_samples
        time_series = self._postprocess_images(images_batch, preprocess_infos)

        # Ensure the requested generation length
        if time_series.shape[1] != generation_length:
            if time_series.shape[1] > generation_length:
                time_series = time_series[:, :generation_length]
            else:
                pad_length = generation_length - time_series.shape[1]
                padding = torch.zeros(num_samples, pad_length)
                time_series = torch.cat([time_series, padding], dim=1)

        return time_series.cpu()
