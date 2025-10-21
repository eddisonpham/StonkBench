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
        """
        Fit GARCH(1,1) independently for each channel.
        
        Args:
            data (np.ndarray): Shape (T, N), already log returns
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)

        T, N = data.shape
        self.series_length = T
        self.params = np.zeros((N, 3), dtype=np.float32)
        self.models = []

        for i in range(N):
            series = data[:, i]
            model = arch_model(series, vol="Garch", p=1, q=1, mean="Zero", dist="normal", rescale=False)
            fitted = model.fit(disp="off")
            self.models.append(fitted)

            omega = fitted.params["omega"]
            alpha = fitted.params["alpha[1]"]
            beta = fitted.params["beta[1]"]
            self.params[i] = [omega, alpha, beta]

            print(f"Channel {i+1}/{N} fitted: omega={omega:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")

        self.fitted = True

    def generate(self, num_samples: int, output_length: int = None, init_values: np.ndarray = None, seed: int = None):
        """
        Generate synthetic series using GARCH(1,1) recursion.
        
        Args:
            num_samples (int): Number of realizations
            output_length (int, optional): Length of series. Defaults to original series length
            init_values (np.ndarray, optional): Initial values (shape (N,) or (num_samples, N))
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            np.ndarray: Shape (num_samples, output_length, N)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before generating data.")

        if output_length is None:
            output_length = self.series_length

        if seed is not None:
            np.random.seed(seed)

        N = self.params.shape[0]
        synthetic = np.zeros((num_samples, output_length, N), dtype=np.float32)

        for i in range(N):
            omega, alpha, beta = self.params[i]
            sigma2 = np.zeros((num_samples, output_length), dtype=np.float32)

            # Initialize first step
            if init_values is not None:
                if init_values.ndim == 1:
                    synthetic[:, 0, i] = init_values[i]
                else:
                    synthetic[:, 0, i] = init_values[:, i]
                sigma2[:, 0] = np.var(synthetic[:, 0, i])
            else:
                sigma2[:, 0] = omega / (1 - alpha - beta)  # unconditional variance
                synthetic[:, 0, i] = np.random.randn(num_samples) * np.sqrt(sigma2[:, 0])

            # Recursive generation for all samples simultaneously
            for t in range(1, output_length):
                sigma2[:, t] = omega + alpha * synthetic[:, t-1, i]**2 + beta * sigma2[:, t-1]
                synthetic[:, t, i] = np.random.randn(num_samples) * np.sqrt(sigma2[:, t])

        return synthetic