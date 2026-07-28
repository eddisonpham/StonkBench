from typing import Protocol, runtime_checkable


@runtime_checkable
class Dataset(Protocol):
    def __iter__(self):
        ...
    def __len__(self) -> int:
        ...
