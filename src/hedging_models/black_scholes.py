"""
Black-Scholes hedging model.
"""

import torch
import numpy as np
from scipy.stats import norm
from typing import Optional

from src.hedging_models.base_hedger import BaseHedgingModel


class BlackScholes(BaseHedgingModel):
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
        d1 = (np.log(S_np / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        return torch.from_numpy(delta).float().to(self.device)
    
    def fit(self, data: torch.Tensor, num_epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = True):
        """Estimate volatility from training data."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")
        
        if data.dim() == 3:
            prices = data[:, :, 0]
        else:
            prices = data
        
        prices = prices.to(self.device).float()
        
        # Estimate volatility from all training data
        self.volatility = self._estimate_volatility(prices)
        
        # Optimize premium using gradient descent
        self.train()
        optimizer = torch.optim.Adam([self.premium], lr=learning_rate)
        
        num_samples = prices.shape[0]
        
        for epoch in range(num_epochs):
            indices = torch.randperm(num_samples)
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_prices = prices[batch_indices]
                
                optimizer.zero_grad()
                
                deltas = self.forward(batch_prices)
                loss = self.compute_loss(batch_prices, deltas)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print(f"Training completed. Final premium: {self.premium.item():.6f}, Volatility: {self.volatility:.6f}")
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """Compute Black-Scholes deltas."""
        if self.volatility is None:
            self.volatility = self._estimate_volatility(prices)

        L = prices.shape[1]
        
        deltas = []
        for t in range(L - 1):
            S_t = prices[:, t]  # Current price
            T_remaining = (L - 1 - t) / (L - 1)  # Normalized time to maturity
            T_remaining = max(T_remaining, 0.01)  # Avoid division by zero
            
            delta_t = self._compute_delta(S_t, self.strike, self.risk_free_rate, 
                                         self.volatility, T_remaining)
            deltas.append(delta_t)
        
        return torch.stack(deltas, dim=1)  # (batch_size, L-1)

