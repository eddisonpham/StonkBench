import torch
import torch.nn as nn

from src.hedging_models.base_hedger import DeepHedgingModel


class FeedforwardLayers(DeepHedgingModel):
    """
    Feed-forward MLP hedger.

    - Input: prices tensor (batch_size, L)
    - Architecture: Flatten -> Dense -> ReLU -> Dense -> ReLU -> Dense -> output (L-1)
    - Output: deltas (batch_size, L-1)
    """

    def __init__(
        self,
        seq_length: int,
        hidden_size: int = 128,
        mlp_layers: int = 5,
        final_scale: float = 1.0,
        strike: float = 1.0,
    ):
        super().__init__(seq_length=seq_length, hidden_size=hidden_size, strike=strike)
        self.mlp_layers = mlp_layers
        self.final_scale = final_scale
        layers = []
        layers.append(nn.Linear(seq_length+1, hidden_size))
        for _ in range(mlp_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_size, seq_length))
        self.mlp = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute deltas.

        prices: (batch_size, L)
        returns deltas: (batch_size, L-1)
        """
        prices = prices.to(self.device).float()
        return self.mlp(prices)

