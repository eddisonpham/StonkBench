import torch
import numpy as np
from xgboost import XGBRegressor

from src.hedging_models.base_hedger import NonDeepHedgingModel


class XGBoost(NonDeepHedgingModel):
    def __init__(self, seq_length: int, strike: float = 1.0, **xgb_params):
        """
        Dynamic XGBoost hedger that adapts to the input sequence length during fit().
        """
        super().__init__(seq_length, strike=strike)
        self.xgb_params = xgb_params
        self.models = []

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        if prices.dim() == 3:
            prices = prices[:, :, 0]
        prices_np = prices.cpu().numpy()
        batch_size, L = prices_np.shape

        if L - 1 != len(self.models):
            raise ValueError(
                f"Number of trained models {len(self.models)} does not match "
                f"sequence length {L} (L-1 models required)"
            )

        deltas_list = []
        for t, model in enumerate(self.models):
            X_t = prices_np[:, : t + 1]
            delta_t = model.predict(X_t)
            # Clip deltas to [0, 1] to avoid invalid hedging positions
            delta_t = np.clip(delta_t, 0.0, 1.0)
            deltas_list.append(delta_t.reshape(batch_size, 1))
        deltas = np.hstack(deltas_list)
        return torch.tensor(deltas, dtype=torch.float32, device=self.device)

    def fit(self, prices: torch.Tensor):
        if prices.dim() == 3:
            prices = prices[:, :, 0]

        _, L = prices.shape
        self.models = [XGBRegressor(**self.xgb_params) for _ in range(L - 1)]

        prices_np = prices.cpu().numpy()
        final_prices = prices[:, -1]
        payoffs = self.compute_payoff(final_prices)
        self.premium = payoffs.mean()

        # Compute target deltas and clip them to [0, 1]
        deltas_targets = []
        for t in range(L - 1):
            S_t = prices[:, t]
            S_tp1 = prices[:, t + 1]
            delta_target = (payoffs - self.premium) / (S_tp1 - S_t + 1e-8)
            delta_target = torch.clamp(delta_target, 0.0, 1.0)
            deltas_targets.append(delta_target.cpu().numpy())

        # Fit models
        for t, model in enumerate(self.models):
            X_t = prices_np[:, : t + 1]
            y_t = deltas_targets[t]
            model.fit(X_t, y_t)
