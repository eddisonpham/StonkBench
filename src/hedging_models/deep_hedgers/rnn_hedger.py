"""
Recurrent Neural Network (RNN) for deep hedging.
"""

import torch
import torch.nn as nn
from src.hedging_models.base_hedger import DeepHedgingModel


class RNN(DeepHedgingModel):
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, num_layers: int = 1):
        super().__init__(seq_length, hidden_size, strike)
        
        self.num_layers = num_layers
        
        # RNN layer: input is price (1 feature), output is hidden state
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer: maps hidden state to delta
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process price sequence through RNN.
        """
        # Input shape: (batch_size, L-1, 1)
        input_prices = prices[:, :-1].unsqueeze(-1)
        
        # Output shape: (batch_size, L-1, hidden_size)
        rnn_out, _ = self.rnn(input_prices)
        
        # Output shape: (batch_size, L-1, 1)
        deltas = self.output(rnn_out).squeeze(-1)  # (batch_size, L-1)
        
        return deltas

