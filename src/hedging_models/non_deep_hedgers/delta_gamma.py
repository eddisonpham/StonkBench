import torch

from src.hedging_models.base_hedger import NonDeepHedgingModel


class DeltaGamma(NonDeepHedgingModel):
    """
    Delta-Gamma hedger using Black-Scholes analytical formulas.
    """

    def __init__(self, seq_length: int, strike: float = 1.0, sigma: float = 0.2, T: float = 1.0):
        super().__init__(seq_length, strike)
        self.sigma = sigma
        self.T = T
        self.r = 0.0

    def black_scholes_d1_d2(self, S: torch.Tensor, t: torch.Tensor):
        """
        Compute d1 and d2 for Black-Scholes formulas.
        S: (batch_size,) current price
        t: (batch_size,) time remaining to maturity
        """
        sigma_sqrt_t = self.sigma * torch.sqrt(t)
        d1 = (torch.log(S / self.strike) + (self.r + 0.5 * self.sigma ** 2) * t) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        return d1, d2

    def compute_delta(self, S: torch.Tensor, t: torch.Tensor):
        """
        Black-Scholes Delta for European call.
        """
        d1, _ = self.black_scholes_d1_d2(S, t)
        delta = torch.distributions.Normal(0, 1).cdf(d1)
        return delta

    def compute_gamma(self, S: torch.Tensor, t: torch.Tensor):
        """
        Black-Scholes Gamma for European call.
        """
        d1, _ = self.black_scholes_d1_d2(S, t)
        gamma = torch.distributions.Normal(0, 1).log_prob(d1).exp() / (S * self.sigma * torch.sqrt(t))
        return gamma

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Compute Delta-Gamma hedging positions at each time step.
        prices: (batch_size, seq_length)
        """
        prices = prices.to(self.device).float()
        batch_size, L = prices.shape
        deltas = torch.zeros((batch_size, L - 1), device=self.device)

        for t in range(L - 1):
            time_remaining = self.T * (L - 1 - t) / (L - 1)  # simple linear time scaling
            S_t = prices[:, t]
            delta_t = self.compute_delta(S_t, torch.full_like(S_t, time_remaining))
            gamma_t = self.compute_gamma(S_t, torch.full_like(S_t, time_remaining))
            # Delta-Gamma hedge
            price_diff = prices[:, t + 1] - prices[:, t]
            deltas[:, t] = delta_t + 0.5 * gamma_t * price_diff

        return deltas

    def fit(self, data: torch.Tensor):
        """
        Analytical hedger, no training needed. Premium set to 0 by default in base; must be set here.
        Set premium to Black-Scholes price of a call option based on average initial price and given params.
        """
        # Compute average initial price for a batch
        S0 = data[:, 0].mean().item()
        K = self.strike
        sigma = self.sigma
        r = self.r
        T = self.T

        S0 = torch.tensor(S0, dtype=torch.float32)
        K = torch.tensor(K, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        T = torch.tensor(T, dtype=torch.float32)

        d1 = (torch.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)
        norm = torch.distributions.Normal(0, 1)
        bs_price = S0 * norm.cdf(d1) - K * torch.exp(-r * T) * norm.cdf(d2)
        self.premium = bs_price.detach().to(self.device)
        print(f"Estimated premium: {self.premium.item():.4f}")
