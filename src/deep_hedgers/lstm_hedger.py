"""
Long Short-Term Memory (LSTM) network for deep hedging.
"""

import torch
import torch.nn as nn
from src.deep_hedgers.base_hedger import BaseDeepHedger


class LSTMDeepHedger(BaseDeepHedger):
    """
    Long Short-Term Memory (LSTM) network for deep hedging.
    
    Architecture:
    - Input: Sequence of open prices S_0, S_1, ..., S_{L-1}
    - LSTM processes the sequence sequentially, capturing long-term dependencies
    - Output: Delta_t for each time step t = 0, ..., L-2
    
    Variables learned:
    - Premium p (scalar)
    - Delta_t for t = 0, ..., L-2 (L-1 values)
    
    Loss function: MSE(X) where X = Final Payoff - Terminal Value
    Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))
    """
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, num_layers: int = 1):
        """
        Args:
            seq_length: Length of the time series (L)
            hidden_size: Size of LSTM hidden state
            strike: Strike price for the call option payoff
            num_layers: Number of LSTM layers
        """
        super().__init__(seq_length, hidden_size, strike)
        
        self.num_layers = num_layers
        
        # LSTM layer: input is price (1 feature), output is hidden state
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer: maps hidden state to delta
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process price sequence through LSTM.
        
        Args:
            prices: Price sequence of shape (batch_size, L)
            
        Returns:
            Deltas of shape (batch_size, L-1)
        """
        batch_size = prices.shape[0]
        
        # Prepare input: prices up to L-1 (we need L-1 deltas)
        # Input shape: (batch_size, L-1, 1)
        input_prices = prices[:, :-1].unsqueeze(-1)
        
        # LSTM forward pass
        # Output shape: (batch_size, L-1, hidden_size)
        lstm_out, _ = self.lstm(input_prices)
        
        # Map hidden states to deltas
        # Output shape: (batch_size, L-1, 1)
        deltas = self.output(lstm_out).squeeze(-1)  # (batch_size, L-1)
        
        return deltas

