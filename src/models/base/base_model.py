"""
Base classes for time series generative models.

This module defines abstract base classes for two main types of generative time series models:

1. StatisticalModel:
    - An abstract interface for statistical generative models, such as GBM, O-U Process, GARCH, block bootstrap, etc.

2. DeepLearningModel:
    - An abstract interface for deep learning generative models using PyTorch.
    - Assumes usage of DataLoader-based training and batch processing.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class StatisticalModel(ABC):
    """
    Base class for statistical generative time series models.

    - Expects as input for fitting: a univariate time series of shape (l,) or (l, 1),
      or a multivariate series of shape (l, c).
    - Outputs generated samples of shape (R, l) for univariate data, or (R, l, c) for multivariate data.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: torch.Tensor, *args, **kwargs) -> None:
        """
        Fits the model parameters using the entire time series.

        Args:
            data (np.ndarray or torch.Tensor): Input data of shape (l,), (l, 1), or (l, c)
            *args, **kwargs: Extra keyword arguments for specialized models.
        """
        pass

    @abstractmethod
    def generate(self, num_samples: int, generation_length: int, seed: int = 42, *args, **kwargs) -> torch.Tensor:
        """
        Generates synthetic time series realizations.

        Args:
            num_samples (int): Number of simulated samples (R).
            generation_length (int): Length of each generated sample.
            seed (int, optional): Random seed for generation. If not given, falls back to the instance seed.
            *args, **kwargs: Extra keyword arguments.

        Returns:
            np.ndarray or torch.Tensor: Generated series of shape (R, l) or (R, l, c)
        """
        pass


class DeepLearningModel(torch.nn.Module, ABC):
    """
    Abstract base class for deep learning time series generative models.

    - Expects as input for fitting: a DataLoader yielding batches of shape (batch_size, l) for univariate series,
      or (batch_size, l, c) for c-channel series (e.g. conditioning inputs or multivariate inputs).
    - Outputs generated samples of shape (R, l) (univariate) or (R, l, c) depending on the model.
    """

    def __init__(self):
        torch.nn.Module.__init__(self)

    @abstractmethod
    def fit(self, data_loader: torch.utils.data.DataLoader, num_epochs: int = 10, *args, **kwargs) -> None:
        """
        Trains the network via a DataLoader with batches.

        Args:
            data_loader (torch.utils.data.DataLoader): Batches of (batch_size, l) for training
            num_epochs (int): Number of training epochs
            valid_loader (torch.utils.data.DataLoader, optional): Batches of (batch_size, l) for validation.
                If provided, model selection will be based on validation loss.
            *args, **kwargs: Extra training keyword arguments.
        """
        pass

    @abstractmethod
    def generate(self, num_samples: int, generation_length: int, seed: int = 42, *args, **kwargs) -> torch.Tensor:
        """
        Generates synthetic samples after training.

        Args:
            num_samples (int): Number of simulated samples (R).
            *args, **kwargs: Optional arguments.

        Returns:
            torch.Tensor: Generated series of shape (R, l) or (R, l, c)
        """
        pass
