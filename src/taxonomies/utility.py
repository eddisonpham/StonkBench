import torch
from typing import Dict, Any
from scipy.stats import spearmanr

from src.hedging_models.base_hedger import DeepHedgingModel, NonDeepHedgingModel
from src.hedging_models.deep_hedgers.feedforward_layers import FeedforwardLayers
from src.hedging_models.deep_hedgers.feedforward_time import FeedforwardTime
from src.hedging_models.deep_hedgers.rnn_hedger import RNN
from src.hedging_models.deep_hedgers.lstm_hedger import LSTM
from src.hedging_models.non_deep_hedgers.black_scholes import BlackScholes
from src.hedging_models.non_deep_hedgers.delta_gamma import DeltaGamma
from src.hedging_models.non_deep_hedgers.linear_regression import LinearRegression
from src.hedging_models.non_deep_hedgers.xgboost import XGBoost
from src.utils.preprocessing_utils import LogReturnTransformation


def log_returns_to_prices(
    log_returns: torch.Tensor,
    initial_prices: torch.Tensor
) -> torch.Tensor:
    """Convert log returns to prices using initial prices."""
    if log_returns.ndim == 1:
        log_returns = log_returns.unsqueeze(0)
    R, L = log_returns.shape
    if initial_prices.shape != (R,):
        raise ValueError(f"initial_prices shape {initial_prices.shape} doesn't match (R,)")
    scaler = LogReturnTransformation()
    prices = torch.zeros((R, L), device=log_returns.device)
    for i in range(R):
        prices_full = scaler.inverse_transform(log_returns[i], initial_prices[i])
        prices[i] = prices_full[1:]
    return prices


def compute_replication_errors(hedger, prices: torch.Tensor) -> torch.Tensor:
    """
    Compute replication errors: R = Final Payoff - Terminal Value
    for each sample path.
    """
    if isinstance(hedger, DeepHedgingModel):
        hedger.eval()
    prices = prices.to(hedger.device).float()
    with torch.no_grad():
        deltas = hedger.forward(prices)
        terminal_values = hedger.compute_terminal_value(prices, deltas)
        final_prices = prices[:, -1]
        payoffs = torch.clamp(final_prices - float(hedger.strike), min=0.0)  # European call
        R = payoffs - terminal_values
    return R


def fit_hedger(
    hedger,
    data: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """Train a hedger (DeepHedgingModel or NonDeepHedgingModel)."""
    if isinstance(hedger, DeepHedgingModel):
        hedger.fit(data, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    elif isinstance(hedger, NonDeepHedgingModel):
        hedger.fit(data)
    else:
        raise ValueError(f"Unknown hedger type: {type(hedger)}")


def summarize_replication_error(R: torch.Tensor) -> Dict[str, float]:
    """Return mean and std of replication error."""
    return {
        'mean': float(R.mean().item()),
        'std': float(R.std().item())
    }


class AugmentedTestingEvaluator:
    """
    Mix synthetic and real training data (50/50) to evaluate hedgers
    based on replication error on the real validation set.
    """
    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        synthetic_train_initial: torch.Tensor = None,
        seq_length: int = None,
        num_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3
    ):
        self.real_train_prices_full = log_returns_to_prices(real_train_log_returns, real_train_initial)
        self.real_val_prices = log_returns_to_prices(real_val_log_returns, real_val_initial)
        if synthetic_train_initial is None:
            mean_initial = float(real_train_initial.mean().item())
            synthetic_train_initial = torch.ones(synthetic_train_log_returns.shape[0], device=real_train_initial.device) * mean_initial
        self.synthetic_train_prices_full = log_returns_to_prices(synthetic_train_log_returns, synthetic_train_initial)
        self.seq_length = seq_length or self.real_train_prices_full.shape[1]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.strike = float(real_train_initial.mean().item())

        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardLayers,
            'Feedforward_Time': FeedforwardTime,
            'RNN': RNN,
            'LSTM': LSTM,
            'BlackScholes': BlackScholes,
            'DeltaGamma': DeltaGamma,
            'LinearRegression': LinearRegression,
            'XGBoost': XGBoost,
        }

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        results = {}
        R_real, R_syn = self.real_train_prices_full.shape[0], self.synthetic_train_prices_full.shape[0]
        R_mixed = min(R_real, R_syn)

        for name, cls in self.hedger_classes.items():
            # Sample balanced subsets for mixed training
            real_idx = torch.randperm(R_real)[:R_mixed]
            syn_idx = torch.randperm(R_syn)[:R_mixed]
            real_subset = self.real_train_prices_full[real_idx]
            syn_subset = self.synthetic_train_prices_full[syn_idx]
            mixed_train = torch.cat([real_subset, syn_subset], dim=0)[torch.randperm(2 * R_mixed)]

            # Train hedgers
            hedger_mixed = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(hedger_mixed, mixed_train, num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

            hedger_real = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(hedger_real, real_subset, num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

            # Compute replication error on validation set
            R_mixed_val = compute_replication_errors(hedger_mixed, self.real_val_prices)
            R_real_val = compute_replication_errors(hedger_real, self.real_val_prices)

            results[name] = {
                'real_train': summarize_replication_error(R_real_val),
                'mixed_train': summarize_replication_error(R_mixed_val)
            }

        return results

class AlgorithmComparisonEvaluator:
    """
    Train all hedgers on real and synthetic data, evaluate on real test set
    using replication error only. Computes Spearman correlation between
    hedger rankings.
    """
    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_test_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_test_initial: torch.Tensor,
        synthetic_train_initial: torch.Tensor = None,
        seq_length: int = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ):
        self.real_train_prices = log_returns_to_prices(real_train_log_returns, real_train_initial)
        self.real_test_prices = log_returns_to_prices(real_test_log_returns, real_test_initial)

        if synthetic_train_initial is None:
            mean_initial = float(real_train_initial.mean().item())
            synthetic_train_initial = torch.ones(
                synthetic_train_log_returns.shape[0],
                device=real_train_initial.device
            ) * mean_initial
        self.synthetic_train_prices = log_returns_to_prices(synthetic_train_log_returns, synthetic_train_initial)

        self.seq_length = seq_length or self.real_train_prices.shape[1]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.strike = float(real_train_initial.mean().item())

        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardLayers,
            'Feedforward_Time': FeedforwardTime,
            'RNN': RNN,
            'LSTM': LSTM,
            'BlackScholes': BlackScholes,
            'DeltaGamma': DeltaGamma,
            'LinearRegression': LinearRegression,
            'XGBoost': XGBoost,
        }

    def evaluate(self) -> Dict[str, Any]:
        real_scores = []
        synthetic_scores = []
        hedger_names = []

        for name, cls in self.hedger_classes.items():
            hedger_names.append(name)

            # Train on real
            hedger_real = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(
                hedger_real,
                self.real_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            R_real_test = compute_replication_errors(hedger_real, self.real_test_prices)
            real_scores.append(R_real_test.mean().item())  # scalar for ranking

            # Train on synthetic
            hedger_syn = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(
                hedger_syn,
                self.synthetic_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            R_syn_test = compute_replication_errors(hedger_syn, self.real_test_prices)
            synthetic_scores.append(R_syn_test.mean().item())  # scalar for ranking

        # Compute Spearman correlation between real-trained and synthetic-trained rankings
        spearman_corr, _ = spearmanr(real_scores, synthetic_scores)

        results = {
            'hedger_names': hedger_names,
            'real_scores': real_scores,
            'synthetic_scores': synthetic_scores,
            'spearman_correlation': spearman_corr
        }
        return results
