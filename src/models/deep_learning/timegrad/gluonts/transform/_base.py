from abc import ABC, abstractmethod


class Transformation(ABC):
    @abstractmethod
    def __call__(self, data):
        return data

    def __add__(self, other):
        if hasattr(other, '__call__'):
            return _Chain([self, other])
        return NotImplemented


class _Chain(Transformation):
    def __init__(self, t):
        self.transformations = list(t)

    def __call__(self, data):
        for t in self.transformations:
            data = t(data)
        return data
