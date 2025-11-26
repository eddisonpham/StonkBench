import torch
import torch.nn as nn
from src.hedging_models.base_hedger import DeepHedgingModel


class FeedforwardTime(DeepHedgingModel):

    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__(seq_length, hidden_size, strike)
        self.hidden = nn.Linear(2, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.to(self.device)
        
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process (S_t, t) pairs through the network.
        """
        batch_size, seq_length = prices.shape
        num_deltas = seq_length - 1
        s_t = prices[:, :-1]
        t = torch.arange(num_deltas, device=prices.device, dtype=prices.dtype)
        t = t.unsqueeze(0).expand(batch_size, num_deltas)
        t_norm = t / (seq_length - 1)
        features = torch.cat([s_t.unsqueeze(-1), t_norm.unsqueeze(-1)], dim=-1)
        flat_features = features.view(-1, 2)
        hidden_out = self.activation(self.hidden(flat_features))
        return self.output(hidden_out).view(batch_size, num_deltas)


