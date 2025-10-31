import torch
import numpy as np
from arch import arch_model
from src.models.base.base_model import ParametricModel


class GARCH11(ParametricModel):
    def __init__(self, length: int, num_channels: int, device='cpu'):
        super().__init__(length, num_channels)
        self.device = device
        self.models = [None] * num_channels
        self.fitted_params = [None] * num_channels

    def fit(self, log_returns, dist='normal'):
        """
        Fit GARCH(1,1) models to each channel of log returns.
        dist: 'normal' or 't' for student-t innovations
        """
        L, N = log_returns.shape
        self.models = []
        self.fitted_params = []

        for i in range(N):
            series = log_returns[:, i]
            am = arch_model(
                series,
                mean='Constant',
                vol='GARCH',
                p=1,
                q=1,
                dist=dist,
                rescale=False
            )
            res = am.fit(disp='off')
            alpha = res.params.get('alpha[1]', res.params.iloc[1])
            beta = res.params.get('beta[1]', res.params.iloc[2])

            self.models.append(am)
            self.fitted_params.append(res)

    def generate(self, num_samples, seq_length=None, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        N = self.num_channels
        L = seq_length if seq_length is not None else self.length
        log_returns_sim = torch.zeros((num_samples, L, N), device=self.device)

        for i in range(N):
            res = self.fitted_params[i]
            params = res.params
            mu = float(params.get('mu', 0.0))
            omega = float(params.get('omega', params.iloc[0]))
            alpha = float(params.get('alpha[1]', params.iloc[1]))
            beta = float(params.get('beta[1]', params.iloc[2]))
            last_sigma2 = float(res.conditional_volatility[-1])**2
            last_r = float(self.models[i].y[-1])
            sigma2 = torch.full((num_samples,), last_sigma2, device=self.device)
            r_prev = torch.full((num_samples,), last_r, device=self.device)
            for t in range(L):
                z = torch.randn(num_samples, device=self.device)
                sigma_t = torch.sqrt(sigma2)
                r_t = mu + sigma_t * z
                log_returns_sim[:, t, i] = r_t
                sigma2 = omega + alpha * r_prev**2 + beta * sigma2
                r_prev = r_t

        return log_returns_sim.cpu()
