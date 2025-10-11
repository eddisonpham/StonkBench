import torch
import os
from abc import ABC, abstractmethod

from src.utils.path_utils import make_sure_path_exist

class BaseGenerativeModel(ABC):
    """
    Abstract base class for all generative models.
    Defines the interface for fitting, generating, saving, and loading models.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def fit(self, data_loader, *args, **kwargs):
        """
        Trains the model on the provided data.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def generate(self, num_samples, *args, **kwargs):
        """
        Generates new time series samples.

        Args:
            num_samples (int): The number of samples to generate.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Generated time series data of shape (num_samples, length, channels).
        """
        pass

    def save_model(self, path):
        """
        Saves the model's state to the specified path.
        For ParametricModel, saves relevant attributes as a dictionary.
        For DeepLearningModel, saves the state_dict (handled by nn.Module inheritance).

        Args:
            path (str): The directory path to save the model.
        """
        make_sure_path_exist(path)
        model_state = {key: value for key, value in self.__dict__.items() if not key.startswith('_') and isinstance(value, (torch.Tensor, float, int, str, bool))}
        torch.save(model_state, os.path.join(path, f'{self.__class__.__name__}_model.pth'))
        print(f"Model state saved to {os.path.join(path, f'{self.__class__.__name__}_model.pth')}")

    def load_model(self, path):
        """
        Loads the model's state from the specified path.

        Args:
            path (str): The directory path from which to load the model.
        """
        state = torch.load(os.path.join(path, f'{self.__class__.__name__}_model.pth'), map_location=self.device)
        for key, value in state.items():
            setattr(self, key, value)
        print(f"Model state loaded from {os.path.join(path, f'{self.__class__.__name__}_model.pth')}")


class ParametricModel(BaseGenerativeModel, ABC):
    """
    Abstract base class for parametric stochastic time series models.
    These models rely on statistical parameter estimation.
    They DO NOT inherit from torch.nn.Module.
    """
    def __init__(self):
        super().__init__()


class DeepLearningModel(torch.nn.Module, BaseGenerativeModel, ABC):
    """
    Abstract base class for non-parametric deep learning generative models (e.g., GANs, VAEs).
    These models are trained using neural networks and inherit from torch.nn.Module.
    """
    def __init__(self):
        BaseGenerativeModel.__init__(self) # Explicitly call BaseGenerativeModel's init
        torch.nn.Module.__init__(self)    # Explicitly call nn.Module's init
