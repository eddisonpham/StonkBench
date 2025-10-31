from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    print("Warning: PyWavelets not available. Please install with: pip install PyWavelets")

from src.models.base.base_model import DeepLearningModel


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim > 1:
            emb = math.log(10000) / (half_dim - 1)
        else:
            emb = 1.0
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=time.dtype) * -emb)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def _safe_group_norm_num_groups(num_channels: int, max_groups: int = 4) -> int:
    """Select the largest number of groups <= max_groups that divides num_channels. FIX"""
    max_groups = min(max_groups, num_channels)
    for g in range(max_groups, 0, -1):
        if (num_channels % g) == 0:
            return g
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        num_groups_in = _safe_group_norm_num_groups(in_channels, max_groups=4)  # FIX
        num_groups_out = _safe_group_norm_num_groups(out_channels, max_groups=4)  # FIX
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb_proj = self.time_mlp(time_emb)
        h = h + time_emb_proj[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class _UNet(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.down1 = _ResidualBlock(in_channels, 64, time_emb_dim)
        self.down2 = _ResidualBlock(64, 128, time_emb_dim)
        self.bottleneck = _ResidualBlock(128, 256, time_emb_dim)
        self.up1 = _ResidualBlock(256 + 128, 128, time_emb_dim)
        self.up2 = _ResidualBlock(128 + 64, 64, time_emb_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(_safe_group_norm_num_groups(64, max_groups=4), 64),  # FIX 
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timestep)
        d1 = self.down1(x, t)
        h, w = d1.shape[2], d1.shape[3]
        d2_size = (max(1, h // 2), max(1, w // 2))
        d2_input = F.adaptive_avg_pool2d(d1, d2_size)
        d2 = self.down2(d2_input, t)
        h2, w2 = d2.shape[2], d2.shape[3]
        b_size = (max(1, h2 // 2), max(1, w2 // 2))
        b_input = F.adaptive_avg_pool2d(d2, b_size)
        b = self.bottleneck(b_input, t)
        u1 = self.up1(torch.cat([F.interpolate(b, size=d2.shape[-2:], mode='bilinear', align_corners=False), d2], dim=1), t)
        u2 = self.up2(torch.cat([F.interpolate(u1, size=d1.shape[-2:], mode='bilinear', align_corners=False), d1], dim=1), t)
        return self.out(u2)


def wavelet_transform_forward(x: torch.Tensor, wavelet: str = 'haar') -> tuple:
    if not PYWAVELETS_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")
    B, L, N = x.shape
    device = x.device
    dtype = x.dtype
    x_np = x.detach().cpu().numpy()
    all_low = []
    all_high = []
    for b in range(B):
        for n in range(N):
            coeffs = pywt.dwt(x_np[b, :, n], wavelet, mode='zero')
            cA, cD = coeffs
            all_low.append(cA)
            all_high.append(cD)
    first_low = all_low[0]
    first_high = all_high[0]
    coef_length = len(first_low)
    low_freq = np.array(all_low).reshape(B, N, coef_length)
    high_freq = np.array(all_high).reshape(B, N, coef_length)
    low_freq = torch.tensor(low_freq, device=device, dtype=dtype)
    high_freq = torch.tensor(high_freq, device=device, dtype=dtype)
    wavelet_img = torch.cat([low_freq, high_freq], dim=2)
    wavelet_img = wavelet_img.unsqueeze(2)
    coeffs_info = {
        'low_shape': (N, coef_length),
        'high_shape': (N, coef_length),
        'original_length': L,
        'wavelet': wavelet
    }
    return wavelet_img, coeffs_info


def wavelet_transform_inverse(wavelet_img: torch.Tensor, coeffs_info: dict, wavelet: Optional[str] = None) -> torch.Tensor:
    if not PYWAVELETS_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")
    if wavelet is None:
        wavelet = coeffs_info.get('wavelet', 'haar')
    B, N, H, W = wavelet_img.shape
    device = wavelet_img.device
    wavelet_img = wavelet_img.squeeze(2)
    coef_length = coeffs_info['low_shape'][1]
    low_freq = wavelet_img[:, :, :coef_length]
    high_freq = wavelet_img[:, :, coef_length:coef_length*2]
    low_np = low_freq.detach().cpu().numpy()
    high_np = high_freq.detach().cpu().numpy()
    reconstructed = []
    for b in range(B):
        batch_recon = []
        for n in range(N):
            recon = pywt.idwt(low_np[b, n], high_np[b, n], wavelet, mode='zero')
            batch_recon.append(recon)
        reconstructed.append(np.stack(batch_recon, axis=0))
    result = torch.tensor(np.stack(reconstructed, axis=0), device=device, dtype=wavelet_img.dtype)
    result = result.transpose(1, 2)
    return result


class TakahashiDiffusion(DeepLearningModel):
    def __init__(
        self,
        length: int,
        num_channels: int,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        wavelet: str = 'haar',
        lr: float = 1e-4,
    ):
        super().__init__(length=length, num_channels=num_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # FIX if base class doesn't set it
        self.num_steps = int(num_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.wavelet = str(wavelet)
        # Create betas on CPU then move to device AFTER model and device defined
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        self.betas = betas.to(self.device)  # FIX
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.model = _UNet(in_channels=num_channels, time_emb_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.to(self.device)  # ensure everything is on correct device

        self.coeffs_info_cache = None
        # Optional: EMA of model parameters for better sampling
        self.ema_decay = 0.9999
        self.ema_params = [p.clone().detach().to(self.device) for p in self.model.parameters()]

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        # FIX: ensure t is on same device and dtype correct
        t = t.to(a.device, dtype=torch.long)
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om_acp_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_acp_t * x_start + sqrt_om_acp_t * noise

    def _p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.model(x, t)
        sqrt_acp_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_om_acp_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        pred_x_start = (x - sqrt_om_acp_t * predicted_noise) / sqrt_acp_t

        posterior_mean_coef1 = self._extract(
            self.sqrt_alphas_cumprod_prev * self.betas / (1.0 - self.alphas_cumprod), t, x.shape
        )
        posterior_mean_coef2 = self._extract(
            self.sqrt_alphas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), t, x.shape
        )
        posterior_mean = posterior_mean_coef1 * pred_x_start + posterior_mean_coef2 * x

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (x.dim() - 1)))  # FIX

        return posterior_mean + nonzero_mask * (torch.sqrt(posterior_variance_t) * noise)

    def fit(
        self,
        data_loader,
        num_epochs: int = 100,
    ):
        self.model.train()
        mse = nn.MSELoss()
        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0
            for batch_idx, real_batch in enumerate(data_loader):
                if isinstance(real_batch, (list, tuple)):
                    real_batch = real_batch[0]
                real = real_batch.to(self.device)
                try:
                    wavelet_img, coeffs_info = wavelet_transform_forward(real, self.wavelet)
                    self.coeffs_info_cache = coeffs_info
                except Exception as e:
                    print(f"Warning: Wavelet transform failed: {e}. Skipping batch.")
                    continue

                batch_size = wavelet_img.shape[0]
                t = torch.randint(0, self.num_steps, (batch_size,), device=self.device).long()
                noise = torch.randn_like(wavelet_img)
                x_t = self._q_sample(wavelet_img, t, noise)
                predicted_noise = self.model(x_t, t)
                loss = mse(noise, predicted_noise)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # FIX: update EMA params
                with torch.no_grad():
                    for ema_p, p in zip(self.ema_params, self.model.parameters()):
                        ema_p.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

                total_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"TakahashiDiffusion epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        seq_length: Optional[int] = None,
        seed: int = 42,
        use_ema: bool = True  # option to use EMA weights
    ) -> torch.Tensor:
        if seq_length is None:
            seq_length = self.length

        if self.coeffs_info_cache is None:
            dummy = torch.zeros(1, seq_length, self.num_channels, device=self.device)
            try:
                _, coeffs_info = wavelet_transform_forward(dummy, self.wavelet)
                self.coeffs_info_cache = coeffs_info
            except Exception as e:
                raise RuntimeError(f"Cannot generate: wavelet transform failed. {e}")
        coeffs_info = self.coeffs_info_cache

        N = self.num_channels
        coef_length = coeffs_info['low_shape'][1]
        shape = (1, N, 1, coef_length * 2)  # generate 1 sample at a time
        generated_list = []

        # option: temporarily load EMA weights
        if use_ema:
            saved = {n: p.clone() for n, p in self.model.named_parameters()}
            for (n, p), ema_p in zip(self.model.named_parameters(), self.ema_params):
                p.data.copy_(ema_p)

        for i_sample in range(num_samples):
            torch.manual_seed(seed + i_sample)  # increment seed for each sample
            x = torch.randn(shape, device=self.device)

            for i in reversed(range(0, self.num_steps)):
                t = torch.full((1,), i, device=self.device, dtype=torch.long)
                x = self._p_sample(x, t)

            try:
                generated = wavelet_transform_inverse(x, coeffs_info, self.wavelet)
            except Exception as e:
                raise RuntimeError(f"Inverse wavelet transform failed: {e}")

            if generated.shape[1] != seq_length:
                if generated.shape[1] > seq_length:
                    generated = generated[:, :seq_length, :]
                else:
                    pad_length = seq_length - generated.shape[1]
                    padding = torch.zeros(1, pad_length, self.num_channels, device=self.device)
                    generated = torch.cat([generated, padding], dim=1)

            generated_list.append(generated)

        # restore original weights if EMA was used
        if use_ema:
            for n, p in self.model.named_parameters():
                p.data.copy_(saved[n])

        # concatenate all generated samples
        return torch.cat(generated_list, dim=0).detach().cpu()
