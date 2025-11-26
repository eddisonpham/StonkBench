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
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output = nn.Linear(hidden_size, 1)
        self.to(self.device)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input shape (batch_size, seq_length), output deltas shape (batch_size, seq_length - 1)
        """
        prices = prices.to(self.device).float()
        input_prices = prices[:, :-1].unsqueeze(-1)  # (batch_size, seq_length-1, 1)
        rnn_out, _ = self.rnn(input_prices)          # (batch_size, seq_length-1, hidden_size)
        deltas = self.output(rnn_out).squeeze(-1)    # (batch_size, seq_length-1)
        return deltas

