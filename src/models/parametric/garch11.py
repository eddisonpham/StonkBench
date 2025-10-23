import numpy as np
from arch import arch_model
from src.models.base.base_model import ParametricModel

class GARCH11(ParametricModel):
    """
    Simplified GARCH(1,1) model fitting each channel independently.
    Assumes input data is already log returns.
    """

    def __init__(self):
        super().__init__()
        self.params = None
        self.models = None
        self.fitted = False
        self.series_length = None

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float64)
        T, N = data.shape
        self.series_length = T
        self.params = np.zeros((N, 3), dtype=np.float64)
        self.models = []

        for i in range(N):
            series = data[:, i]
            scale_factor = 10 / np.std(series)
            series_scaled = series * scale_factor
            model = arch_model(series_scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            fitted = model.fit(disp="off")
            omega = fitted.params["omega"] / (scale_factor ** 2)
            alpha = fitted.params["alpha[1]"]
            beta  = fitted.params["beta[1]"]
            self.params[i] = [omega, alpha, beta]
            print(f"Channel {i+1}/{N} fitted: omega={omega:.6e}, alpha={alpha:.6e}, beta={beta:.6e}")

        self.fitted = True


    def generate(self, num_samples: int, output_length: int = None, init_values: np.ndarray = None, seed: int = None):
        """
        Generate synthetic multivariate log-return time series using fitted GARCH(1,1) parameters.

        Args:
            num_samples: Number of independent synthetic series to generate.
            output_length: Length of each generated series. Defaults to fitted series length.
            init_values: Optional initial values to start the series. Shape (N,) or (num_samples, N).
            seed: Random seed for reproducibility.

        Returns:
            synthetic: np.ndarray of shape (num_samples, output_length, N)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before generating data.")

        if output_length is None:
            output_length = self.series_length

        if seed is not None:
            np.random.seed(seed)

        L, N = self.params.shape[0], self.params.shape[0]  # N channels
        synthetic = np.zeros((num_samples, output_length, N), dtype=np.float64)

        for i in range(N):
            omega, alpha, beta = self.params[i]
            sigma2 = np.zeros((num_samples, output_length), dtype=np.float64)

            # Initialize first value
            if init_values is not None:
                if init_values.ndim == 1:
                    synthetic[:, 0, i] = init_values[i]
                else:
                    synthetic[:, 0, i] = init_values[:, i]

                sigma2[:, 0] = omega + alpha * synthetic[:, 0, i]**2
                sigma2[:, 0] = np.maximum(sigma2[:, 0], 1e-12)

            else:
                # Use unconditional variance for log returns
                sigma2[:, 0] = omega / max(1e-8, 1 - alpha - beta)
                synthetic[:, 0, i] = np.random.randn(num_samples) * np.sqrt(sigma2[:, 0])

            # GARCH recursion
            for t in range(1, output_length):
                sigma2[:, t] = omega + alpha * synthetic[:, t-1, i]**2 + beta * sigma2[:, t-1]
                sigma2[:, t] = np.maximum(sigma2[:, t], 1e-12)
                synthetic[:, t, i] = np.random.randn(num_samples) * np.sqrt(sigma2[:, t])

        return synthetic
