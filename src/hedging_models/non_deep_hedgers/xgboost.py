import torch
import numpy as np
from xgboost import XGBRegressor

from src.hedging_models.base_hedger import NonDeepHedgingModel


class XGBoost(NonDeepHedgingModel):
    def __init__(self, seq_length: int, strike: float = 1.0, **xgb_params):
        super().__init__(seq_length, strike=strike)
        self.xgb_params = xgb_params
        self.models = []
        self.cached_prefixes = None

    # ----------------------------
    # FAST FORWARD PASS
    # ----------------------------
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        if prices.dim() == 3:
            prices = prices[:, :, 0]  # (B, L)
        prices_np = prices.cpu().numpy()
        B, L = prices_np.shape

        if len(self.models) != L - 1:
            raise RuntimeError(
                f"Expected {L-1} trained models, found {len(self.models)}"
            )

        # Vectorized prefix-cache creation (only once per forward)
        prefixes = [
            prices_np[:, : (t + 1)] for t in range(L - 1)
        ]

        # Predict all deltas in one vectorized loop
        deltas = np.zeros((B, L - 1), dtype=np.float32)

        for t, model in enumerate(self.models):
            deltas[:, t] = model.predict(prefixes[t])

        deltas = np.clip(deltas, 0.0, 1.0)
        return torch.tensor(deltas, device=self.device)

    # ----------------------------
    # FAST FIT
    # ----------------------------
    def fit(self, prices: torch.Tensor):
        if prices.dim() == 3:
            prices = prices[:, :, 0]

        B, L = prices.shape
        prices_np = prices.cpu().numpy()

        # Create L–1 models
        self.models = [XGBRegressor(**self.xgb_params) for _ in range(L - 1)]

        # Payoff & premium
        final_prices = prices[:, -1]
        payoffs = self.compute_payoff(final_prices)
        self.premium = payoffs.mean()

        payoffs_np = payoffs.cpu().numpy()

        # --------------------------
        # Precompute price diffs ONCE
        # --------------------------
        S_t = prices_np[:, :-1]       # (B, L-1)
        S_tp1 = prices_np[:, 1:]      # (B, L-1)
        denom = (S_tp1 - S_t + 1e-8)

        # Compute all target deltas vectorized
        delta_targets = (payoffs_np[:, None] - float(self.premium)) / denom
        delta_targets = np.clip(delta_targets, 0.0, 1.0)

        # --------------------------
        # Precompute all feature prefixes (fast)
        # --------------------------
        prefixes = [
            prices_np[:, : (t + 1)] for t in range(L - 1)
        ]

        # --------------------------
        # Fit all models
        # --------------------------
        for t, model in enumerate(self.models):
            X_t = prefixes[t]
            y_t = delta_targets[:, t]
            model.fit(X_t, y_t)
