"""
Feedforward Neural Network with time step as input for deep hedging.
"""

import torch
import torch.nn as nn
from src.hedging_models.base_hedger import BaseHedgingModel


class FeedforwardTime(BaseHedgingModel):

    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__(seq_length, hidden_size, strike)
        
        # Input: (price, time_step) = 2 features
        # Output: 1 delta value
        self.hidden = nn.Linear(2, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process (S_t, t) pairs through the network.
        """
        batch_size = prices.shape[0]
        num_deltas = self.seq_length - 1
        
        # Create time steps: [0, 1, 2, ..., L-2]
        time_steps = torch.arange(num_deltas, device=prices.device, dtype=prices.dtype)
        time_steps = time_steps.unsqueeze(0).expand(batch_size, -1)  # (batch_size, L-1)
        
        # Normalize time steps to [0, 1] range
        time_steps_norm = time_steps / (self.seq_length - 1)
        
        # Get prices for time steps 0 to L-2
        input_prices = prices[:, :-1]  # (batch_size, L-1)
        
        # Combine price and time step: (batch_size, L-1, 2)
        input_features = torch.stack([input_prices, time_steps_norm], dim=-1)
        
        # Reshape for batch processing: (batch_size * (L-1), 2)
        input_flat = input_features.view(-1, 2)
        
        # Forward through network
        hidden_out = self.activation(self.hidden(input_flat))  # (batch_size * (L-1), hidden_size)
        deltas_flat = self.output(hidden_out)  # (batch_size * (L-1), 1)
        
        # Reshape back: (batch_size, L-1)
        deltas = deltas_flat.view(batch_size, num_deltas)
        
        return deltas

