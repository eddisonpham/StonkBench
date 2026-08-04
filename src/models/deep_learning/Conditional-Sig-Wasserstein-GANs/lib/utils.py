import pickle

import numpy as np
import torch


def sample_indices(dataset_size, batch_size, device=None):
    """Sample indices on the same device as the tensor they index.

    The original vendor helper unconditionally moved indices to CUDA, which
    makes CPU validation/smoke runs fail and can create a needless device
    mismatch when the calibrated signature target is still on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = torch.from_numpy(
        np.random.choice(dataset_size, size=batch_size, replace=False)
    ).to(device=device, dtype=torch.long)
    # numpy choice is intentionally retained: torch multinomial/choice is slow
    # for this small, repeatedly sampled index set.
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()
