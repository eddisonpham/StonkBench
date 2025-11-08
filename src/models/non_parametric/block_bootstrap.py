import torch
import numpy as np

class BlockBootstrap:
    def __init__(self, block_size: int = 13, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.block_size = block_size
        self.seed = seed
        self.log_returns = None

    def fit(self, log_returns: torch.Tensor):
        self.log_returns = log_returns

    def generate(self, num_samples: int, generation_length: int):
        total_time_steps = self.log_returns.shape[0]
        num_blocks = int(np.ceil(generation_length / self.block_size))

        samples = torch.zeros((num_samples, generation_length),
                              dtype=self.log_returns.dtype)

        for sample_idx in range(num_samples):
            idxs = []
            for _ in range(num_blocks):
                start_idx = torch.randint(0, total_time_steps - self.block_size + 1, (1,)).item()
                block_idxs = list(range(start_idx, start_idx + self.block_size))
                idxs.extend(block_idxs)
            idxs = idxs[:generation_length]
            samples[sample_idx, :] = self.log_returns[idxs]

        return samples


