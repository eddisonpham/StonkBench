import torch
import torch.nn as nn
from src.hedging_models.base_hedger import DeepHedgingModel


class FeedforwardLayers(DeepHedgingModel):
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__(seq_length, hidden_size, strike)
        
        # Build network with L-1 hidden layers
        layers = []
        input_size = 1  # Input is single price S_t
        
        for i in range(seq_length - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process each price S_t through the network to get Delta_t.
        """
        batch_size = prices.shape[0]
        
        # Input: prices[:, :-1] shape (batch_size, L-1)
        # Reshape to (batch_size * (L-1), 1)
        input_prices = prices[:, :-1].unsqueeze(-1)  # (batch_size, L-1, 1)
        input_flat = input_prices.reshape(-1, 1)  # (batch_size * (L-1), 1)
        
        # Output shape: (batch_size * (L-1), 1)
        output_flat = self.network(input_flat)
        
        # Reshape back to (batch_size, L-1)
        deltas = output_flat.reshape(batch_size, self.seq_length - 1)
        
        return deltas

