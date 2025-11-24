"""
Black-Scholes hedging model.
"""

import torch
import numpy as np
from scipy.stats import norm
from typing import Optional

from src.hedging_models.base_hedger import NonDeepHedgingModel


class BlackScholes(NonDeepHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, risk_free_rate: float = 0.0):
        super().__init__(seq_length, hidden_size, strike)
        self.risk_free_rate = risk_free_rate
        self.volatility = None
        self.time_to_maturity = 1.0
    
    def _estimate_volatility(self, prices: torch.Tensor) -> float:
        """Estimate volatility from log returns."""
        prices_np = prices.cpu().numpy()
        log_returns = np.diff(np.log(prices_np), axis=1)
        volatility = float(np.std(log_returns))
        return max(volatility, 1e-6)
    
    def _compute_delta(self, S: torch.Tensor, K: float, r: float, sigma: float, T: float) -> torch.Tensor:
        """Compute Black-Scholes delta."""
        S_np = S.cpu().numpy()
        sqrt_T = np.sqrt(max(T, 1e-6))
        d1 = (np.log(S_np / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        delta = norm.cdf(d1)
        return torch.from_numpy(delta).float().to(self.device)
    
    def _compute_premium(self, S0: torch.Tensor, K: float, r: float, sigma: float, T: float) -> float:
        """Compute Black-Scholes option premium analytically."""
        S0_np = S0.cpu().numpy() if isinstance(S0, torch.Tensor) else S0
        S0_mean = float(np.mean(S0_np))
        sqrt_T = np.sqrt(max(T, 1e-6))
        d1 = (np.log(S0_mean / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        premium = S0_mean * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return float(premium)
    
    def fit(self, data: torch.Tensor, verbose: bool = True):
        """
        Estimate volatility and compute premium using analytical Black-Scholes formula.
        No gradient-based optimization is used.
        """
        if data.dim() == 3:
            prices = data[:, :, 0]
        else:
            prices = data
        
        prices = prices.to(self.device).float()
        
        # Estimate volatility from all training data
        self.volatility = self._estimate_volatility(prices)
        
        # Compute premium analytically using Black-Scholes formula
        # Use initial prices (first time step) for premium calculation
        S0 = prices[:, 0]
        self.premium = torch.tensor(
            self._compute_premium(S0, self.strike, self.risk_free_rate, 
                                 self.volatility, self.time_to_maturity),
            device=self.device
        )
        
        if verbose:
            print(f"Training completed. Premium: {self.premium.item():.6f}, Volatility: {self.volatility:.6f}")
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """Compute Black-Scholes deltas."""
        if self.volatility is None:
            self.volatility = self._estimate_volatility(prices)

        L = prices.shape[1]
        
        deltas = []
        for t in range(L - 1):
            S_t = prices[:, t]  # Current price
            T_remaining = (L - 1 - t) / (L - 1)  # Normalized time to maturity
            
            delta_t = self._compute_delta(S_t, self.strike, self.risk_free_rate, 
                                         self.volatility, T_remaining)
            deltas.append(delta_t)
        
        return torch.stack(deltas, dim=1)  # (batch_size, L-1)

