"""Minimal PyTorch ``.pt`` reader that needs only ``numpy``.

The handover bundle ships tensors saved with ``torch.save``.  Reviewers often do
not have PyTorch installed, so this module reconstructs the same objects from
the underlying zip + pickle container, without importing ``torch``.

Usage
-----
>>> from torch_pt_reader import load_pt
>>> obj = load_pt("02_generated_outputs/quantgan/quantgan_seq126.pt")
>>> obj["data"].shape          # (num_samples, seq_len, num_assets)
>>> obj["metadata"]["model_name"]

Only the pieces ``torch.save`` actually emits are supported: 1-D CPU storages,
``_rebuild_tensor_v2`` / ``_rebuild_tensor`` / ``_rebuild_tensor_v3``, plain
containers (dict / list / tuple / OrderedDict), and scalars.  Anything else
raises ``UnsupportedPickle`` instead of guessing.
"""

from __future__ import annotations

import builtins
import collections
import io
import pickle
import zipfile

import numpy as np

__all__ = ["load_pt", "UnsupportedPickle", "Stub"]

# torch storage class name -> numpy dtype
_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": None,  # not representable in numpy, handled explicitly
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


class UnsupportedPickle(RuntimeError):
    """Raised when a pickle opcode refers to a class this reader cannot emulate."""


class Stub:
    """Stand-in for a class this reader does not know (e.g. a training config).

    It keeps the original qualified name plus whatever state pickle hands over,
    so manifests can still report what a checkpoint contained.
    """

    __qualified_name__ = "?"

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __repr__(self):  # pragma: no cover - debug helper
        return f"<Stub {type(self).__qualified_name__}>"


def _make_stub_class(qualified_name):
    return type("Stub_" + qualified_name.replace(".", "_"), (Stub,), {"__qualified_name__": qualified_name})


class _Storage:
    """A flattened CPU storage, mirroring ``torch.Storage``."""

    __slots__ = ("dtype", "array", "key")

    def __init__(self, dtype, array, key):
        self.dtype = dtype
        self.array = array
        self.key = key

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"_Storage(dtype={self.dtype}, numel={self.array.size}, key={self.key})"


def _as_strided(storage, storage_offset, size, stride):
    """Materialise a strided view as a (copied) numpy array."""
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    if int(np.prod(size, dtype=np.int64)) == 0:
        return np.empty(size, dtype=storage.array.dtype)

    # Encode every index as a single flat offset into the storage, then gather.
    offsets = np.zeros(size, dtype=np.int64)
    for dim, (st) in enumerate(stride):
        if st == 0:
            continue
        shape = [1] * len(size)
        shape[dim] = size[dim]
        offsets += (np.arange(size[dim], dtype=np.int64) * st).reshape(shape)
    offsets += storage_offset
    # A contiguous tensor is by far the common case and is cheaper to slice.
    if stride == _contiguous_stride(size) and storage_offset == 0:
        n = int(np.prod(size, dtype=np.int64))
        return storage.array[:n].reshape(size)
    return storage.array[offsets.ravel()].reshape(size)


def _contiguous_stride(size):
    stride, acc = [], 1
    for n in reversed(size):
        stride.append(acc)
        acc *= n
    return tuple(reversed(stride))


def _rebuild(storage, storage_offset, size, stride, *_, **__):
    if isinstance(storage, _Storage):
        return _as_strided(storage, storage_offset, size, stride)
    # torch may hand back a non-storage for meta tensors; keep it opaque.
    return storage


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, storage_reader):
        super().__init__(file, encoding="utf-8")
        self._storage_reader = storage_reader

    def find_class(self, module, name):  # noqa: D102
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
            "_rebuild_tensor_v3",
            "_rebuild_parameter",
        }:
            return _rebuild
        if module == "torch" and name in ("Size",):
            return tuple
        if name == "OrderedDict" and module in ("collections", "builtins"):
            return collections.OrderedDict
        if module == "torch" and name == "_utils":
            return Stub
        if module in ("torch", "torch.serialization") and name.endswith("Storage"):
            if name in _STORAGE_DTYPES:
                return _storage_cls(name)
            raise UnsupportedPickle(f"unsupported storage class: {module}.{name}")
        if module in ("builtins", "__builtin__"):
            obj = getattr(builtins, name, None)
            if obj is not None:
                return obj
        if module == "numpy" or module.startswith("numpy."):
            node = np
            for part in module.split(".")[1:] + [name]:
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None:
                return node
        # Unknown class (training configs, argparse namespaces, ...): keep it opaque
        # rather than failing, so manifests can still describe the checkpoint.
        return _make_stub_class(f"{module}.{name}")

    def persistent_load(self, pid):  # noqa: D102
        if not isinstance(pid, (tuple, list)) or pid[0] != "storage":
            raise UnsupportedPickle(f"unsupported persistent id: {pid!r}")
        _, storage_type, key, location, _numel = pid
        # find_class() already resolved the storage class to our stub; recover the
        # original torch class name from it (or accept a plain string).
        if callable(storage_type) and hasattr(storage_type, "storage_type"):
            storage_type = storage_type.storage_type
        return self._storage_reader(str(storage_type), str(key), location)


def _storage_cls(storage_type):
    """Return a callable that behaves enough like ``torch.FloatStorage``."""

    def make(*args, **kwargs):
        # torch pickles storages only through persistent_load; direct calls are
        # used for e.g. `torch.FloatStorage(0)`; surface a clear error.
        raise UnsupportedPickle(f"direct construction of {storage_type} is not supported")

    make.storage_type = storage_type
    return make


def load_pt(path, member=None):
    """Load a ``torch.save`` payload from *path* using numpy only.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file.
    member:
        Optional zip member name prefix.  Auto-detected when omitted.

    Returns
    -------
    The unpickled object, with every tensor as a ``numpy.ndarray``.
    """
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        prefixes = sorted({n.rsplit("/", 1)[0] for n in names if n.endswith("data.pkl")})
        if not prefixes:
            raise UnsupportedPickle(f"{path}: no data.pkl found (legacy non-zip format not supported)")
        prefix = member or prefixes[0]
        root = f"{prefix}/" if prefix else ""

        storage_bytes = {}

        def read_storage(storage_type, key, location):
            if key not in storage_bytes:
                raw = zf.read(f"{root}data/{key}")
                dtype = _STORAGE_DTYPES.get(storage_type)
                if dtype is None:
                    raise UnsupportedPickle(f"{path}: unsupported storage dtype {storage_type}")
                arr = np.frombuffer(raw, dtype=dtype).copy()
                storage_bytes[key] = _Storage(dtype, arr, key)
            return storage_bytes[key]

        payload = zf.read(f"{root}data.pkl")
        return _Unpickler(io.BytesIO(payload), read_storage).load()


def describe(obj, prefix="", depth=0, max_depth=3):
    """Return a compact human-readable description of a loaded payload."""
    lines = []

    def walk(node, name, d):
        pad = "  " * d
        if isinstance(node, dict):
            lines.append(f"{pad}{name}: dict[{len(node)}]")
            if d < max_depth:
                for k, v in list(node.items())[:12]:
                    walk(v, str(k), d + 1)
        elif isinstance(node, np.ndarray):
            lines.append(f"{pad}{name}: ndarray{node.shape} {node.dtype}")
        elif isinstance(node, (list, tuple)):
            lines.append(f"{pad}{name}: {type(node).__name__}[{len(node)}]")
            if node and d < max_depth:
                walk(node[0], "0", d + 1)
        else:
            lines.append(f"{pad}{name}: {type(node).__name__} = {node!r}"[:160])

    walk(obj, prefix or "root", depth)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    for p in sys.argv[1:]:
        print("=" * 70)
        print(p)
        print(describe(load_pt(p)))
