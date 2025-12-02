import torch
import torch.nn as nn

from src.hedging_models.base_hedger import DeepHedgingModel


class FeedforwardTime(DeepHedgingModel):
    """
    MLP that takes the whole price series + a normalized time index for each
    delta position and outputs (batch, L-1) deltas.
    """

    def __init__(self, seq_length: int, hidden_size: int = 64, strike: float = 1.0):
        super().__init__(seq_length=seq_length, hidden_size=hidden_size, strike=strike)

        self.seq_length = seq_length

        # Input = full price path (L) + time index (1)
        self.hidden = nn.Linear(seq_length + 1, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        prices: (batch_size, L)
        returns deltas: (batch_size, L-1)
        """
        prices = prices.to(self.device).float()
        batch_size, seq_length = prices.shape
        num_deltas = seq_length - 1

        # Normalized time indices: [0, 1, ..., L-2] / (L-1)
        t = torch.linspace(0, (num_deltas - 1) / (seq_length - 1), num_deltas,
                           device=prices.device, dtype=prices.dtype)

        # Expand:
        # prices_exp: (B, L-1, L)
        # t_exp:      (B, L-1, 1)
        prices_exp = prices.unsqueeze(1).expand(batch_size, num_deltas, seq_length)
        t_exp = t.view(1, num_deltas, 1).expand(batch_size, num_deltas, 1)

        # Concatenate to (B, L-1, L+1)
        x = torch.cat([prices_exp, t_exp], dim=-1)

        # Flatten for linear layer: (B*(L-1), L+1)
        x = x.reshape(batch_size * num_deltas, seq_length + 1)

        # MLP
        h = self.activation(self.hidden(x))
        out = self.output(h)  # shape: (B*(L-1), 1)

        # Reshape to (B, L-1)
        return out.view(batch_size, num_deltas)




