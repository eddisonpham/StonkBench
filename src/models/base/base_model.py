"""
Base classes for time series generative models.

This module defines abstract base classes for two main types of generative time series models:

1. ParametricModel:
    - An abstract interface for statistical (parametric) generative models, such as GBM, O-U Process, GARCH, etc.
    - Assumes direct fitting to arrays/tensors representing time series data.
    - Provides methods to fit model parameters, generate synthetic series, and save/load model state.

2. DeepLearningModel:
    - An abstract interface for non-parametric (deep learning) generative models using PyTorch.
    - Assumes usage of DataLoader-based training and batch processing.
    - Provides methods for training (fit), sample generation, and saving/loading PyTorch model weights.

Custom model classes should inherit from one of these base classes.
"""

import torch
from abc import ABC, abstractmethod


class ParametricModel(ABC):
    """
    Abstract base class for parametric (statistical) generative time series models.

    - Expects as input for fitting: a single time series of shape (l, N) where
      l is the sequence length and N is the number of channels/features.
    - Outputs generated samples of shape (R, l, N) where R is the number of simulated realizations.
    """

    def __init__(self, length: int, num_channels: int):
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def fit(self, data, *args, **kwargs):
        """
        Fits the model parameters using the entire time series.

        Args:
            data (np.ndarray or torch.Tensor): Input data of shape (l, N)
            *args, **kwargs: Extra keyword arguments for specialized models.
        """
        pass

    @abstractmethod
    def generate(self, num_samples, seq_length=None, init_values=None, seed=42, *args, **kwargs):
        """
        Generates synthetic time series realizations.

        Args:
            num_samples (int): Number of simulated samples (R).
            *args, **kwargs: Extra keyword arguments.

        Returns:
            np.ndarray or torch.Tensor: Generated series of shape (R, l, N)
        """
        pass

class DeepLearningModel(torch.nn.Module, ABC):
    """
    Abstract base class for non-parametric (deep learning) time series generative models.

    - Expects as input for fitting: a DataLoader yielding batches of shape (batch_size, l, N)
      suitable for gradient-based training.
    - Outputs generated samples of shape (R, l, N).
    """

    def __init__(self, length: int, num_channels: int):
        self.length = int(length)
        self.num_channels = int(num_channels)
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def fit(self, data_loader, num_epochs=10, *args, **kwargs):
        """
        Trains the network via a DataLoader with batches.

        Args:
            data_loader (torch.utils.data.DataLoader): Batches of (batch_size, l, N)
            *args, **kwargs: Extra training keyword arguments.
        """
        pass

    @abstractmethod
    def generate(self, num_samples, seq_length=None, seed=42, *args, **kwargs):
        """
        Generates synthetic samples after training.

        Args:
            num_samples (int): Number of simulated samples (R).
            *args, **kwargs: Optional arguments.

        Returns:
            torch.Tensor: Generated series of shape (R, l, N)
        """
        pass
