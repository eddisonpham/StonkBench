"""
Random Forest hedging model.
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.hedging_models.base_hedger import BaseHedgingModel


class RandomForest(BaseHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0,
                 n_estimators: int = 100, max_depth: int = 10):
        super().__init__(seq_length, hidden_size, strike)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []  # One model per time step
    
    def fit(self, data: torch.Tensor, num_epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = True):
        """Train Random Forest models for each time step."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")
        
        if data.dim() == 3:
            prices = data[:, :, 0]
        else:
            prices = data
        
        prices_np = prices.cpu().numpy()
        R, L = prices_np.shape
        
        # For each time step, train a model to predict optimal delta
        # We'll use the actual payoff as target (simplified approach)
        self.models = []
        
        for t in range(L - 1):
            # Features: current price, prices up to t, time to maturity
            X = []
            y = []
            
            for i in range(R):
                features = [
                    prices_np[i, t],  # Current price
                    prices_np[i, :t+1].mean() if t > 0 else prices_np[i, 0],  # Mean price so far
                    (L - 1 - t) / (L - 1),  # Time to maturity
                    self.strike,  # Strike price
                ]
                X.append(features)
                
                # Target: approximate delta using price change
                if t < L - 1:
                    price_change = prices_np[i, t+1] - prices_np[i, t]
                    # Simple target: normalized price change
                    target = np.clip(price_change / (prices_np[i, t] + 1e-6), -1, 1)
                    y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            self.models.append(model)
        
        # Optimize premium
        prices = prices.to(self.device).float()
        self.train()
        optimizer = torch.optim.Adam([self.premium], lr=learning_rate)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            deltas = self.forward(prices)
            loss = self.compute_loss(prices, deltas)
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        if verbose:
            print(f"Training completed. Final premium: {self.premium.item():.6f}")
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """Predict deltas using Random Forest models."""
        if len(self.models) == 0:
            # If not trained, return zeros
            batch_size = prices.shape[0]
            return torch.zeros(batch_size, self.seq_length - 1, device=self.device)
        
        prices_np = prices.cpu().numpy()
        batch_size, L = prices_np.shape
        
        deltas = []
        for t in range(L - 1):
            X = []
            for i in range(batch_size):
                features = [
                    prices_np[i, t],
                    prices_np[i, :t+1].mean() if t > 0 else prices_np[i, 0],
                    (L - 1 - t) / (L - 1),
                    self.strike,
                ]
                X.append(features)
            
            X = np.array(X)
            delta_t = self.models[t].predict(X)
            deltas.append(torch.from_numpy(delta_t).float().to(self.device))
        
        return torch.stack(deltas, dim=1)  # (batch_size, L-1)

