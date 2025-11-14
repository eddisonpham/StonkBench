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
    
    Args:
        x: Input time series of shape (L,)
        
    Returns:
        expanded: Expanded series of length 2^n
        original_start: Start index of original data in expanded array
        original_end: End index of original data in expanded array
    """
    L = len(x)
    n = math.ceil(math.log2(L))
    target_length = 2 ** n
    
    if L == target_length:
        return x, 0, L
    
    # Calculate padding needed on each side
    total_padding = target_length - L
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad
    
    # Mirror reflection: reflect the edges
    left_reflection = np.flip(x[:left_pad]) if left_pad > 0 else np.array([])
    right_reflection = np.flip(x[-right_pad:]) if right_pad > 0 else np.array([])
    
    expanded = np.concatenate([left_reflection, x, right_reflection])
    original_start = left_pad
    original_end = left_pad + L
    
    return expanded, original_start, original_end


def power_transform_and_normalize(x: np.ndarray, power: float = 1.0) -> Tuple[np.ndarray, float, float]:
    """
    Apply power transformation and normalization.
    
    Args:
        x: Input time series
        power: Power index p (default 1.0 for no transformation)
        
    Returns:
        transformed: Transformed and normalized series
        mean_val: Mean of original series (for inverse transform)
        std_val: Std of original series (for inverse transform)
    """
    mean_val = np.mean(x)
    std_val = np.std(x)
    
    if std_val < 1e-8:
        return (x - mean_val), mean_val, std_val
    
    centered = x - mean_val
    
    if abs(power - 1.0) < 1e-6:
        # No power transformation
        transformed = centered / std_val
    else:
        # Apply power transformation: sign(x) * |x|^(1/p)
        sign = np.sign(centered)
        abs_val = np.abs(centered)
        transformed = sign * (abs_val ** (1.0 / power)) / std_val
    
    return transformed, mean_val, std_val


def winsorize(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Winsorize outliers by clipping values beyond threshold.
    
    Args:
        x: Input time series
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        winsorized: Winsorized series
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
    
    Args:
        x: Input time series of length 2^n
        wavelet: Wavelet type (default 'haar')
        
    Returns:
        image: Image array where row k contains coefficients of order k
        coeffs_info: Dictionary with coefficient information
    """
    L = len(x)
    n = int(math.log2(L))
    
    # Perform full DWT decomposition
    coeffs = pywt.wavedec(x, wavelet, mode='zero', level=n-1)
    
    # coeffs[0] is approximation (lowest level), coeffs[1:] are details
    # For Haar: coeffs[0] has 1 value, coeffs[1] has 1, coeffs[2] has 2, etc.
    
    # Build image: each row corresponds to a decomposition level
    image_rows = []
    row_length = L  # All rows have same length
    
    for i, coeff in enumerate(coeffs):
        num_coeffs = len(coeff)
        # Split row into num_coeffs segments, fill each uniformly
        segment_length = row_length // num_coeffs
        row = np.repeat(coeff, segment_length)
        # Handle remainder if row_length not divisible by num_coeffs
        if len(row) < row_length:
            row = np.pad(row, (0, row_length - len(row)), mode='edge')
        image_rows.append(row)
    
    # Stack rows to form image
    image = np.stack(image_rows, axis=0)  # Shape: (n_levels, L)
    
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
    
    # Extract coefficients from image rows
    coeffs = []
    for i in range(n_levels):
        row = image[i]
        num_coeffs = coeff_lengths[i]
        segment_length = len(row) // num_coeffs
        # Extract first value from each segment
        coeff = row[::segment_length][:num_coeffs]
        coeffs.append(coeff)
    
    # Reconstruct using inverse DWT
    reconstructed = pywt.waverec(coeffs, wavelet, mode='zero')
    
    return reconstructed


class TakahashiDiffusion(DeepLearningModel):
    """
    Takahashi Diffusion model for generating synthetic time series.
    
    Follows specifications in takahashi.md:
    - Preprocessing: mirror expansion, power transform, normalization, winsorization
    - Wavelet transform with coefficient-to-image mapping
    - Uses HuggingFace's DDPM for diffusion
    
    NOTE: This implementation is for UNIVARIATE time series (log returns only).
    It uses 1 channel (grayscale) images, NOT 3-channel RGB images.
    The markdown mentions RGB for the general case (log returns, spreads, volume),
    but this implementation only handles log returns (1 channel).
    
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
        
        self.length = length  # Will be inferred from data if None
        self.num_channels = int(num_channels)
        self.num_steps = int(num_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.wavelet = str(wavelet)
        self.power_transform = float(power_transform)
        self.winsorize_threshold = float(winsorize_threshold)
        self.lr = float(lr)
        self.device = torch.device(device)
        
        # UNet2DModel from HuggingFace
        # For grayscale images (univariate), in_channels=1
        self.unet = None
        self.scheduler = None
        self.optimizer = None
        
        # Preprocessing statistics (will be computed during fit)
        self.data_mean = None
        self.data_std = None
        self.expansion_info = None  # (original_start, original_end)
        self.coeffs_info_cache = None

    def _init_model(self, image_height: int, image_width: int):
        """Initialize UNet2DModel and DDPMScheduler."""
        if self.unet is None:
            self.unet = UNet2DModel(
                sample_size=(image_height, image_width),
                in_channels=1,  # Grayscale for univariate
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "UpBlock2D",
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

    def _preprocess_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess batch according to takahashi.md specifications.
        
        Returns:
            images: Preprocessed images ready for DDPM (batch_size, 1, H, W)
            preprocess_info: Dictionary with preprocessing parameters for inverse
        """
        batch_np = batch.detach().cpu().numpy()
        batch_size = batch_np.shape[0]
        
        processed_images = []
        preprocess_infos = []
        
        for i in range(batch_size):
            x = batch_np[i]  # (seq_length,)
            
            # 1. Mirror expansion to power of 2
            expanded, orig_start, orig_end = mirror_expand_to_power_of_2(x)
            
            # 2. Power transformation + normalization
            transformed, mean_val, std_val = power_transform_and_normalize(
                expanded, self.power_transform
            )
            
            # 3. Winsorization
            winsorized = winsorize(transformed, self.winsorize_threshold)
            
            # 4. DWT to image coefficients
            image, coeffs_info = dwt_to_image_coefficients(winsorized, self.wavelet)
            
            # Convert to torch tensor and add channel dimension
            # Image shape: (n_levels, L) -> (1, n_levels, L) for grayscale
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            
            processed_images.append(image_tensor)
            preprocess_infos.append({
                'original_start': orig_start,
                'original_end': orig_end,
                'mean': mean_val,
                'std': std_val,
                'coeffs_info': coeffs_info
            })
        
        # Stack into batch: (batch_size, 1, H, W)
        images = torch.stack(processed_images, dim=0).to(self.device)
        
        # Store preprocessing info (use first sample's info as template)
        if self.coeffs_info_cache is None:
            self.coeffs_info_cache = preprocess_infos[0]['coeffs_info']
            self.expansion_info = (preprocess_infos[0]['original_start'], preprocess_infos[0]['original_end'])
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
            image = images[i, 0].detach().cpu().numpy()  # (H, W)
            info = preprocess_info[i] if isinstance(preprocess_info, list) else preprocess_info
            
            # 1. Inverse DWT
            expanded = image_coefficients_to_dwt(image, info['coeffs_info'], self.wavelet)
            
            # 2. Reverse winsorization (already done, no inverse needed)
            
            # 3. Reverse normalization and power transform
            # transformed = (x - mean) / std, so x = transformed * std + mean
            # For power transform: if y = sign(x) * |x|^(1/p), then x = sign(y) * |y|^p
            denormalized = expanded * info['std'] + info['mean']
            
            if abs(self.power_transform - 1.0) > 1e-6:
                sign = np.sign(denormalized)
                abs_val = np.abs(denormalized)
                denormalized = sign * (abs_val ** self.power_transform)
            
            # 4. Remove mirror expansion
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
            x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        return x

    def fit(self, data_loader, num_epochs: int = 100, *args, **kwargs):
        """
        Train Takahashi Diffusion model.
        
        Args:
            data_loader: DataLoader providing batches of shape (batch_size, seq_length)
            num_epochs: Number of training epochs
        """
        # Infer length from first batch if not set
        if self.length is None:
            first_batch = next(iter(data_loader))
            self.length = first_batch.shape[-1] if first_batch.dim() >= 1 else len(first_batch)
            print(f"Inferred sequence length: {self.length}")
        
        # Process first batch to get image dimensions
        first_batch = next(iter(data_loader))
        images, _ = self._preprocess_batch(first_batch)
        _, _, img_h, img_w = images.shape
        print(f"Image dimensions: {img_h} x {img_w}")
        
        # Initialize model
        self._init_model(img_h, img_w)
        
        self.unet.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0
            
            for batch in data_loader:
                # Preprocess to images
                images, _ = self._preprocess_batch(batch)
                
                # Sample noise
                noise = torch.randn_like(images)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=self.device
                ).long()
                
                # Add noise to images
                noisy_images = self.scheduler.add_noise(images, noise, timesteps)
                
                # Predict noise
                noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            if batch_count > 0 and (epoch + 1) % max(1, num_epochs // 10) == 0:
                avg_loss = total_loss / batch_count
                print(f"TakahashiDiffusion epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        self.unet.eval()

    @torch.no_grad()
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
        if self.unet is None:
            raise RuntimeError("Model must be trained before generating samples.")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create dummy batch to get image dimensions and preprocessing info
        dummy_batch = torch.zeros(1, generation_length)
        dummy_images, dummy_info = self._preprocess_batch(dummy_batch)
        _, _, img_h, img_w = dummy_images.shape
        
        # Generate images using DDPM
        generated_images = []
        preprocess_infos = []
        
        for i in range(num_samples):
            torch.manual_seed(seed + i)
            
            # Start with random noise
            image = torch.randn((1, 1, img_h, img_w), device=self.device)
            
            # Denoising loop
            self.scheduler.set_timesteps(self.num_steps)
            for t in self.scheduler.timesteps:
                # Predict noise
                noise_pred = self.unet(image, t, return_dict=False)[0]
                
                # Update image
                image = self.scheduler.step(noise_pred, t, image, return_dict=False)[0]
            
            generated_images.append(image)
            preprocess_infos.append(dummy_info[0])  # Use same preprocessing info
        
        # Stack images
        images_batch = torch.cat(generated_images, dim=0)  # (num_samples, 1, H, W)
        
        # Postprocess to recover time series
        time_series = self._postprocess_images(images_batch, preprocess_infos)
        
        # Adjust length if needed
        if time_series.shape[1] != generation_length:
            if time_series.shape[1] > generation_length:
                time_series = time_series[:, :generation_length]
            else:
                pad_length = generation_length - time_series.shape[1]
                padding = torch.zeros(num_samples, pad_length, device=self.device)
                time_series = torch.cat([time_series, padding], dim=1)
        
        return time_series.cpu()
