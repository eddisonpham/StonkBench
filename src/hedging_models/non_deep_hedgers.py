"""
Non-deep hedging models (traditional methods) that work with the same interface as deep hedgers.
"""

import torch
import numpy as np
from scipy.stats import norm
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb

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


class DeltaGamma(BaseHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, 
                 risk_free_rate: float = 0.0, gamma_weight: float = 0.1):
        super().__init__(seq_length, hidden_size, strike)
        self.risk_free_rate = risk_free_rate
        self.gamma_weight = gamma_weight
        self.volatility = None
        self.time_to_maturity = 1.0
    
    def _estimate_volatility(self, prices: torch.Tensor) -> float:
        """Estimate volatility from log returns."""
        prices_np = prices.cpu().numpy()
        log_returns = np.diff(np.log(prices_np), axis=1)
        volatility = float(np.std(log_returns))
        return max(volatility, 1e-6)
    
    def _compute_delta_gamma(self, S: torch.Tensor, K: float, r: float, 
                            sigma: float, T: float) -> tuple:
        """Compute Black-Scholes delta and gamma."""
        S_np = S.cpu().numpy()
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S_np / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S_np * sigma * sqrt_T)
        
        return (torch.from_numpy(delta).float().to(self.device),
                torch.from_numpy(gamma).float().to(self.device))
    
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
        self.volatility = self._estimate_volatility(prices)
        
        # Optimize premium
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
        """Compute delta-gamma adjusted hedging positions."""
        if self.volatility is None:
            self.volatility = self._estimate_volatility(prices)
        
        batch_size = prices.shape[0]
        L = prices.shape[1]
        
        deltas = []
        prev_gamma = None
        
        for t in range(L - 1):
            S_t = prices[:, t]
            T_remaining = max((L - 1 - t) / (L - 1), 0.01)
            
            delta_t, gamma_t = self._compute_delta_gamma(S_t, self.strike, 
                                                         self.risk_free_rate,
                                                         self.volatility, T_remaining)
            
            # Adjust delta based on gamma change
            if prev_gamma is not None:
                gamma_change = gamma_t - prev_gamma
                delta_t = delta_t + self.gamma_weight * gamma_change
            
            deltas.append(delta_t)
            prev_gamma = gamma_t
        
        return torch.stack(deltas, dim=1)  # (batch_size, L-1)


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


class LinearRegression(BaseHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__(seq_length, hidden_size, strike)
        self.models = []  # One model per time step
    
    def fit(self, data: torch.Tensor, num_epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = True):
        """Train Linear Regression models for each time step."""
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
            
            model = LinearRegression()
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
        """Predict deltas using Linear Regression models."""
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


class XGBoost(BaseHedgingModel):
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0,
                 n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__(seq_length, hidden_size, strike)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.xgb_learning_rate = learning_rate
        self.models = []
    
    def fit(self, data: torch.Tensor, num_epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, verbose: bool = True):
        """Train XGBoost models for each time step."""
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
            
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.xgb_learning_rate,
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
        """Predict deltas using XGBoost models."""
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

