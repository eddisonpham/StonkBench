"""
Base class for deep hedging models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseHedgingModel(nn.Module, ABC):
    """
    Base class for hedging models.
    
    All hedging models learn:
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
        batch_size = prices.shape[0]
        
        # Price differences: S_{t+1} - S_t
        price_diffs = prices[:, 1:] - prices[:, :-1]  # (batch_size, L-1)
        
        # Terminal value = premium + sum(Delta_t * (S_{t+1} - S_t))
        terminal_value = self.premium.squeeze() + torch.sum(deltas * price_diffs, dim=1)
        
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

