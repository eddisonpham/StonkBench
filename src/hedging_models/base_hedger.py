"""
Base classes for hedging models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class DeepHedgingModel(nn.Module, ABC):
    """
    Base class for deep hedging models.
    
    All deep hedging models learn:
    - Premium p (scalar): Initial premium paid for the option
    - Hedging strategy Delta (L-1,): Hedging positions at each time step
    
    Loss function: Mean Squared Error of X = Final Payoff - Terminal Value
    where Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))
    
    For a call option: Payoff(S_T) = max(S_T - K, 0)
    """
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.strike = strike
        
        self.premium = nn.Parameter(torch.zeros(1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def compute_payoff(self, final_prices: torch.Tensor) -> torch.Tensor:
        """
        Compute the payoff at maturity (call option).
        """
        return torch.clamp(final_prices.squeeze() - self.strike, min=0.0) # European call option payoff
    
    def compute_terminal_value(
        self, 
        prices: torch.Tensor, 
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the terminal value of the hedged portfolio.
        
        Terminal Value = p + sum_{t=0}^{L-2} Delta_t * (S_{t+1} - S_t)
        """
        # Price differences: S_{t+1} - S_t
        price_diffs = prices[:, 1:] - prices[:, :-1]  # (batch_size, L-1)
        
        # Sum of delta-weighted price changes: (batch_size,)
        delta_weighted_changes = torch.sum(deltas * price_diffs, dim=1)
        
        # Terminal value = premium + sum(Delta_t * (S_{t+1} - S_t))
        # Premium is broadcast across batch_size
        premium_value = self.premium.squeeze() if self.premium.dim() > 0 else self.premium
        terminal_value = premium_value + delta_weighted_changes
        
        return terminal_value
    
    def compute_loss(
        self, 
        prices: torch.Tensor, 
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the hedging loss.
        
        Loss = MSE(X) where X = Payoff - Terminal Value
        """
        final_prices = prices[:, -1]  # (batch_size,)
        payoff = self.compute_payoff(final_prices)  # (batch_size,)
        terminal_value = self.compute_terminal_value(prices, deltas)
        X = payoff - terminal_value
        loss = torch.mean(X ** 2)
        return loss
    
    @abstractmethod
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute hedging deltas.
        """
        pass
    
    def fit(
        self, 
        data: torch.Tensor, 
        num_epochs: int = 100, 
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """
        Train the deep hedging model.
        """
        # Ensure data is a torch tensor
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")
        
        # Extract open channel (first channel, index 0)
        if data.dim() == 3:
            prices = data[:, :, 0]  # (R, L) - using open channel
        else:
            prices = data  # Assume already (R, L)
        
        prices = prices.to(self.device).float()
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        num_samples = prices.shape[0]
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_prices = prices[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass: compute deltas
                deltas = self.forward(batch_prices)
                
                # Compute loss
                loss = self.compute_loss(batch_prices, deltas)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print(f"Training completed. Final premium: {self.premium.item():.6f}")
    
    def evaluate(self, prices: torch.Tensor) -> dict:
        """
        Evaluate the hedging strategy on given prices.
        """
        self.eval()
        
        if not isinstance(prices, torch.Tensor):
            raise TypeError(f"prices must be torch.Tensor, got {type(prices)}")
        if prices.dim() == 3:
            prices = prices[:, :, 0]
        prices = prices.to(self.device).float()
        
        with torch.no_grad():
            deltas = self.forward(prices)
            terminal_values = self.compute_terminal_value(prices, deltas)
            final_prices = prices[:, -1]
            payoffs = self.compute_payoff(final_prices)
            X = payoffs - terminal_values
            
            results = {
                'premium': self.premium.item(),
                'mean_X': X.mean().item(),
                'std_X': X.std().item(),
                'mse_X': (X ** 2).mean().item(),
                'mean_payoff': payoffs.mean().item(),
                'mean_terminal_value': terminal_values.mean().item(),
            }
        
        return results


class NonDeepHedgingModel(ABC):
    """
    Base class for non-deep learning hedging models.
    
    These models don't use PyTorch neural networks or gradient-based optimization.
    They use traditional ML methods (scikit-learn, LightGBM, XGBoost, etc.) or
    analytical formulas (Black-Scholes, etc.).
    
    All hedging models learn:
    - Premium p (scalar): Initial premium paid for the option
    - Hedging strategy Delta (L-1,): Hedging positions at each time step
    
    Loss function: Mean Squared Error of X = Final Payoff - Terminal Value
    where Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))
    
    For a call option: Payoff(S_T) = max(S_T - K, 0)
    """
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.strike = strike
        
        # Premium as a simple float (not a PyTorch parameter)
        self.premium = torch.zeros(1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_payoff(self, final_prices: torch.Tensor) -> torch.Tensor:
        """
        Compute the payoff at maturity (call option).
        """
        return torch.clamp(final_prices.squeeze() - self.strike, min=0.0)  # European call option payoff
    
    def compute_terminal_value(
        self, 
        prices: torch.Tensor, 
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the terminal value of the hedged portfolio.
        
        Terminal Value = p + sum_{t=0}^{L-2} Delta_t * (S_{t+1} - S_t)
        """
        # Price differences: S_{t+1} - S_t
        price_diffs = prices[:, 1:] - prices[:, :-1]  # (batch_size, L-1)
        
        # Sum of delta-weighted price changes: (batch_size,)
        delta_weighted_changes = torch.sum(deltas * price_diffs, dim=1)
        
        # Terminal value = premium + sum(Delta_t * (S_{t+1} - S_t))
        # Premium is broadcast across batch_size
        if isinstance(self.premium, torch.Tensor):
            premium_value = self.premium.squeeze() if self.premium.dim() > 0 else self.premium
        else:
            premium_value = float(self.premium)
        terminal_value = premium_value + delta_weighted_changes
        
        return terminal_value
    
    def compute_loss(
        self, 
        prices: torch.Tensor, 
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the hedging loss.
        
        Loss = MSE(X) where X = Payoff - Terminal Value
        """
        final_prices = prices[:, -1]  # (batch_size,)
        payoff = self.compute_payoff(final_prices)  # (batch_size,)
        terminal_value = self.compute_terminal_value(prices, deltas)
        X = payoff - terminal_value
        loss = torch.mean(X ** 2)
        return loss
    
    @abstractmethod
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute hedging deltas.
        """
        pass
    
    @abstractmethod
    def fit(
        self, 
        data: torch.Tensor, 
        verbose: bool = True
    ):
        """
        Train the hedging model.
        Non-deep learning models should implement their own training logic.
        """
        pass
    
    def evaluate(self, prices: torch.Tensor) -> dict:
        """
        Evaluate the hedging strategy on given prices.
        """
        if prices.dim() == 3:
            prices = prices[:, :, 0]
        prices = prices.to(self.device).float()
        
        with torch.no_grad():
            deltas = self.forward(prices)
            terminal_values = self.compute_terminal_value(prices, deltas)
            final_prices = prices[:, -1]
            payoffs = self.compute_payoff(final_prices)
            X = payoffs - terminal_values
            
            results = {
                'premium': self.premium.item() if isinstance(self.premium, torch.Tensor) else float(self.premium),
                'mean_X': X.mean().item(),
                'std_X': X.std().item(),
                'mse_X': (X ** 2).mean().item(),
                'mean_payoff': payoffs.mean().item(),
                'mean_terminal_value': terminal_values.mean().item(),
            }
        
        return results

