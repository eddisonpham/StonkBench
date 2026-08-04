from abc import ABC, abstractmethod


class TimeFeature(ABC):
    @abstractmethod
    def __call__(self, index):
        raise NotImplementedError
