from contextlib import contextmanager


class _Env:
    @staticmethod
    @contextmanager
    def _let(**kwargs):
        yield


env = _Env()
