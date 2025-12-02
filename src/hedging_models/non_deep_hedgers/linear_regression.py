import torch
from sklearn.linear_model import LinearRegression as LR

from src.hedging_models.base_hedger import NonDeepHedgingModel


class LinearRegression(NonDeepHedgingModel):
    """
    Hedging model using traditional linear regression.
    """
    def __init__(self, seq_length: int, strike: float = 1.0):
        super().__init__(seq_length=seq_length, strike=strike)
        self.model = LR()
        self.deltas = torch.zeros(seq_length - 1)  # L-1 deltas
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Return hedging deltas for a batch of price paths.
        """
        batch_size = prices.shape[0]
        deltas_batch = self.deltas.unsqueeze(0).expand(batch_size, -1)
        return deltas_batch.to(prices.device).float()
    
    def fit(self, data: torch.Tensor):
        """
        Train linear regression hedger.
        data: (num_paths, seq_length)
        """
        if data.dim() == 3:
            data = data[:, :, 0]

        data = data.cpu().numpy()
        X = data[:, 1:] - data[:, :-1]  # shape (num_paths, seq_length - 1)
        S_T = data[:, -1]
        y = torch.clamp(torch.tensor(S_T) - self.strike, min=0.0).numpy()
        self.model.fit(X, y)
        self.deltas = torch.tensor(self.model.coef_, dtype=torch.float32)
        delta_pnl = X @ self.deltas.numpy()
        self.premium = torch.tensor((y - delta_pnl).mean(), dtype=torch.float32)
        print(f"LinearRegression premium set to: {self.premium.item():.4f}")