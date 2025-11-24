import torch
import torch.nn as nn
from src.hedging_models.base_hedger import DeepHedgingModel


class LSTM(DeepHedgingModel):
    
    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0, num_layers: int = 1):
        super().__init__(seq_length, hidden_size, strike)
        
        self.num_layers = num_layers
        
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
        """
        input_prices = prices[:, :-1].unsqueeze(-1)
        lstm_out, _ = self.lstm(input_prices)
        deltas = self.output(lstm_out).squeeze(-1)
        return deltas

