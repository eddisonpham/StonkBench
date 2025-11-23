"""
LightGBM hedging model.
"""

import torch
import numpy as np
import lightgbm as lgb

from src.hedging_models.base_hedger import BaseHedgingModel


class LightGBM(BaseHedgingModel):
    """
    LightGBM hedger that learns delta from price sequences.
    """
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0,
                 n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__(seq_length, hidden_size, strike)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lgb_learning_rate = learning_rate
        self.models = []
    
    def fit(self, data: torch.Tensor, num_epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = True):
        """Train LightGBM models for each time step."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")
        
        if data.dim() == 3:
            prices = data[:, :, 0]
        else:
            prices = data
        
        prices_np = prices.cpu().numpy()
        R, L = prices_np.shape
        
        self.models = []
        
        for t in range(L - 1):
            X = []
            y = []
            
            for i in range(R):
                features = [
                    prices_np[i, t],
                    prices_np[i, :t+1].mean() if t > 0 else prices_np[i, 0],
                    (L - 1 - t) / (L - 1),
                    self.strike,
                ]
                X.append(features)
                
                if t < L - 1:
                    price_change = prices_np[i, t+1] - prices_np[i, t]
                    target = np.clip(price_change / (prices_np[i, t] + 1e-6), -1, 1)
                    y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            train_data = lgb.Dataset(X, label=y)
            model = lgb.train(
                {
                    'objective': 'regression',
                    'metric': 'mse',
                    'num_leaves': 31,
                    'learning_rate': self.lgb_learning_rate,
                    'max_depth': self.max_depth,
                    'verbose': -1
                },
                train_data,
                num_boost_round=self.n_estimators
            )
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
        """Predict deltas using LightGBM models."""
        if len(self.models) == 0:
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

