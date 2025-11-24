import torch
from torch import Tensor
from torch.distributions.normal import Normal
from typing import Optional

from src.hedging_models.base_hedger import NonDeepHedgingModel


class DeltaGamma(NonDeepHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, 
                 risk_free_rate: float = 0.0, gamma_weight: float = 0.1, time_to_maturity: float = 1.0):
        super().__init__(seq_length, hidden_size, strike)
        self.risk_free_rate = risk_free_rate
        self.gamma_weight = gamma_weight
        self.volatility: Optional[float] = None
        self.time_to_maturity = time_to_maturity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _estimate_volatility(self, prices: Tensor) -> float:
        log_returns = torch.log(prices[:, 1:] / prices[:, :-1])
        volatility = float(log_returns.std().item())
        return max(volatility, 1e-6)
    
    def _compute_premium(self, S0: Tensor, K: float, r: float, sigma: float, T: float) -> Tensor:
        sqrt_T = torch.sqrt(torch.tensor(max(T, 1e-6), device=self.device))
        d1 = (torch.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        std_normal = Normal(0, 1)
        premium = S0 * std_normal.cdf(d1) - K * torch.exp(-r * T) * std_normal.cdf(d2)
        return premium
    
    def fit(self, data: Tensor, verbose: bool = True):
        if data.dim() == 3:
            prices = data[:, :, 0]
        else:
            prices = data
        prices = prices.to(self.device).float()
        self.volatility = self._estimate_volatility
        S0 = prices[:, 0].mean()
        self.premium = self._compute_premium(S0, self.strike, self.risk_free_rate, 
                                             self.volatility, self.time_to_maturity)
        if verbose:
            print(f"[DeltaGamma] Premium: {self.premium.mean().item():.6f}, Volatility: {self.volatility:.6f}")
    
    def forward(self, prices: Tensor) -> Tensor:
        batch_size, L = prices.shape
        prices = prices.to(self.device).float()
        if self.volatility is None:
            self.volatility = self._estimate_volatility(prices)
        
        std_normal = Normal(0, 1)
        deltas = []
        prev_gamma = None
        
        for t in range(L - 1):
            S_t = prices[:, t]
            T_remaining = self.time_to_maturity * max((L - 1 - t) / (L - 1), 1e-6)
            sqrt_T = torch.sqrt(torch.tensor(T_remaining, device=self.device))
            
            d1 = (torch.log(S_t / self.strike) + (self.risk_free_rate + 0.5 * self.volatility**2) * T_remaining) / (self.volatility * sqrt_T)
            
            delta_t = std_normal.cdf(d1)
            gamma_t = std_normal.log_prob(d1).exp() / (S_t * self.volatility * sqrt_T)
            
            if prev_gamma is not None:
                gamma_change = gamma_t - prev_gamma
                delta_t = torch.clamp(delta_t + self.gamma_weight * gamma_change, 0.0, 1.0)
            
            deltas.append(delta_t)
            prev_gamma = gamma_t
        
        return torch.stack(deltas, dim=1)  # shape: (batch_size, L-1)
