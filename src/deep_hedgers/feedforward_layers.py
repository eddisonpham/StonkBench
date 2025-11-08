"""
Feedforward Neural Network with L-1 hidden layers for deep hedging.
"""

import torch
import torch.nn as nn
from src.deep_hedgers.base_hedger import BaseDeepHedger


class FeedforwardDeepHedger(BaseDeepHedger):
    """
    Feedforward Neural Network with L-1 hidden layers.
    
    Architecture:
    - Input at time t: S_t (open price)
    - L-1 hidden layers, each producing a hidden state
    - Output: Delta_t for each time step t = 0, ..., L-2
    
    Variables learned:
    - Premium p (scalar)
    - Delta_t for t = 0, ..., L-2 (L-1 values)
    
    Loss function: MSE(X) where X = Final Payoff - Terminal Value
    Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))
    """
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        """
        Args:
            seq_length: Length of the time series (L)
            hidden_size: Size of hidden layers
            strike: Strike price for the call option payoff
        """
        super().__init__(seq_length, hidden_size, strike)
        
        # L-1 hidden layers: each layer processes the input sequentially
        # Build network with L-1 hidden layers
        layers = []
        input_size = 1  # Input is single price S_t
        
        for i in range(seq_length - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer: produces a single delta value
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process each price S_t through the network to get Delta_t.
        
        The network has L-1 hidden layers, and each price S_t (for t=0 to L-2)
        is processed independently through the network to produce Delta_t.
        
        Args:
            prices: Price sequence of shape (batch_size, L)
            
        Returns:
            Deltas of shape (batch_size, L-1)
        """
        batch_size = prices.shape[0]
        
        # Process each time step's price independently
        # Input: prices[:, :-1] shape (batch_size, L-1)
        # Reshape to (batch_size * (L-1), 1)
        input_prices = prices[:, :-1].unsqueeze(-1)  # (batch_size, L-1, 1)
        input_flat = input_prices.view(-1, 1)  # (batch_size * (L-1), 1)
        
        # Forward through network: produces one delta per price
        # Output shape: (batch_size * (L-1), 1)
        output_flat = self.network(input_flat)
        
        # Reshape back to (batch_size, L-1)
        deltas = output_flat.view(batch_size, self.seq_length - 1)
        
        return deltas

