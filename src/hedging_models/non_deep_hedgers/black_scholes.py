import torch
from scipy.stats import norm

from src.hedging_models.base_hedger import NonDeepHedgingModel


class BlackScholes(NonDeepHedgingModel):
    """
    Black-Scholes analytical hedging model for European call options.
    
    Attributes:
        sigma: Volatility of the underlying asset.
        r: Risk-free interest rate (default 0 for simplicity).
    """
    def __init__(self, seq_length: int, strike: float = 1.0, sigma: float = None, r: float = 0.0):
        super().__init__(seq_length, strike)
        self.sigma = sigma
        self.r = r  # risk-free rate

    def fit(self, data: torch.Tensor):
        """
        Estimate volatility sigma if not provided.
        `data` should be (batch_size, seq_length) of historical prices.
        """
        if self.sigma is None:
            log_returns = torch.log(data[:, 1:] / data[:, :-1])
            self.sigma = log_returns.std().item() * (self.seq_length ** 0.5)
        
        S0 = data[:, 0].mean()
        self.premium = torch.tensor([self.black_scholes_price(S0)], device=data.device)
        print(f"Estimated premium: {self.premium.item():.4f}")

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Compute Black-Scholes delta at each time step.
        prices: (batch_size, seq_length)
        Returns:
            deltas: (batch_size, seq_length-1)
        """
        batch_size, L = prices.shape
        deltas = torch.zeros(batch_size, L-1, device=prices.device)

        for t in range(L-1):
            S_t = prices[:, t]
            T_minus_t = (L - 1 - t) / (L - 1)  # time to maturity fraction
            deltas[:, t] = self.black_scholes_delta(S_t, T_minus_t)
        
        return deltas

    def black_scholes_price(self, S: torch.Tensor, K: float = None, T: float = 1.0) -> torch.Tensor:
        """Compute Black-Scholes call option price. Fully tensor-based."""
        K = K if K is not None else self.strike
        sigma, r = self.sigma, self.r
        if sigma is None:
            raise ValueError("Sigma (volatility) must be set before pricing.")

        S = torch.as_tensor(S, dtype=torch.float32)
        T = torch.as_tensor(T, dtype=torch.float32)
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        K = torch.as_tensor(K, dtype=torch.float32)

        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)
        price = S * torch.tensor(norm.cdf(d1.cpu().numpy()), device=S.device) - K * torch.exp(-r * T) * torch.tensor(norm.cdf(d2.cpu().numpy()), device=S.device)
        return price

    def black_scholes_delta(self, S: torch.Tensor, T: float) -> torch.Tensor:
        """Compute Black-Scholes delta for a call option. Fully tensor-based."""
        sigma, r, K = self.sigma, self.r, self.strike
        S = torch.as_tensor(S, dtype=torch.float32)
        T = torch.as_tensor(T, dtype=torch.float32)
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        K = torch.as_tensor(K, dtype=torch.float32)

        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        delta = torch.tensor(norm.cdf(d1.cpu().numpy()), device=S.device)
        return delta
