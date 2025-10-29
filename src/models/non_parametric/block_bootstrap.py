import torch
import numpy as np

class BlockBootstrap:
    def __init__(self, block_size: int = 5, device: str = "cpu"):
        """
        independent_channels: If True, blocks are sampled independently for each channel.
        circular: If True, blocks wrap around the end of the series for edge cases.
        """
        self.block_size = block_size
        self.device = device
        self.log_returns = None

    def fit(self, log_returns: torch.Tensor):
        if not torch.is_tensor(log_returns):
            log_returns = torch.tensor(log_returns, dtype=torch.float32, device=self.device)
        self.log_returns = log_returns.detach().to(self.device)

    def generate(self, num_samples: int, seq_length: int = None, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = self.log_returns
        T, N = X.shape
        L = seq_length if seq_length is not None else T
        bs = self.block_size
        num_blocks = int(np.ceil(L / bs))

        samples = torch.zeros((num_samples, L, N), dtype=X.dtype, device=X.device)

        for r in range(num_samples):
            for c in range(N):
                idxs = []
                for _ in range(num_blocks):
                    start_idx = np.random.randint(0, T - bs + 1)
                    block_idxs = list(range(start_idx, start_idx + bs))
                    idxs.extend(block_idxs)
                idxs = idxs[:L]
                samples[r, :, c] = X[idxs, c]
        return samples
