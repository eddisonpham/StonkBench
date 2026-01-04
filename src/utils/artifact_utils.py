import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import numpy as np


REQUIRED_METADATA_KEYS = {
    "model_name",
    "model_type",
    "sequence_length",
    "num_samples",
    "seed",
    "preprocessing_cfg",
    "preprocessing_hash",
    "timestamp",
}


def _canonicalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable, sorted version of the config for hashing."""
    def _convert(value: Any):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return _convert(cfg)


def compute_preprocessing_hash(cfg: Dict[str, Any]) -> str:
    """Stable hash of the preprocessing config."""
    canonical = _canonicalize_cfg(cfg)
    payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_artifact(data: torch.Tensor, metadata: Dict[str, Any]) -> None:
    """Validate tensor shape/dtype and metadata contract."""
    if not isinstance(data, torch.Tensor):
        raise ValueError("Artifact data must be a torch.Tensor.")
    if data.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"Artifact tensor dtype must be float, got {data.dtype}.")
    if not torch.isfinite(data).all():
        raise ValueError("Artifact tensor contains non-finite values.")

    missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise ValueError(f"Metadata missing required keys: {missing}")

    seq_len = int(metadata["sequence_length"])
    if data.ndim != 2:
        raise ValueError(f"Artifact tensor must be 2D (num_samples, L); got ndim={data.ndim}.")
    if data.shape[1] != seq_len:
        raise ValueError(
            f"Sequence length mismatch: tensor has {data.shape[1]}, metadata says {seq_len}."
        )


def save_artifact(
    data: torch.Tensor,
    metadata: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Validate and save an artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validate_artifact(data, metadata)

    payload = {"data": data.cpu(), "metadata": metadata}
    torch.save(payload, output_path)
    return output_path


def load_artifact(path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load artifact from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "data" not in payload or "metadata" not in payload:
        raise ValueError(f"Malformed artifact at {path}: expected dict with data+metadata.")
    data = payload["data"]
    metadata = payload["metadata"]
    validate_artifact(data, metadata)
    return data, metadata


def stitch_sequences(base_data: torch.Tensor, target_length: int, seed: int = 42) -> torch.Tensor:
    """
    Create sequences of target_length by stitching/tiling base-length outputs.
    Deterministic given the seed.
    
    Args:
        base_data: 2D tensor of shape (num_samples, base_length)
        target_length: Desired output sequence length
        seed: Random seed for deterministic stitching
        
    Returns:
        2D tensor of shape (num_samples, target_length)
    """
    if base_data.ndim != 2:
        raise ValueError("base_data must be 2D (num_samples, base_length).")
    num_samples, base_length = base_data.shape

    if target_length == base_length:
        return base_data
    if target_length < base_length:
        return base_data[:, :target_length]

    repeats = int(np.ceil(target_length / base_length))
    rng = np.random.default_rng(seed)
    stitched = []
    for i in range(num_samples):
        # Randomly pick a starting offset to reduce periodic artifacts
        offset = rng.integers(0, base_length)
        tiled = base_data[i]
        if offset > 0:
            tiled = torch.roll(tiled, shifts=-offset)
        expanded = tiled.repeat(repeats)[:target_length]
        stitched.append(expanded.unsqueeze(0))
    return torch.cat(stitched, dim=0)


def default_metadata(
    model_name: str,
    model_type: str,
    sequence_length: int,
    num_samples: int,
    seed: int,
    preprocessing_cfg: Dict[str, Any],
    generator_version: str = "v1",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a metadata dictionary with defaults."""
    meta = {
        "model_name": model_name,
        "model_type": model_type,
        "sequence_length": int(sequence_length),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "preprocessing_cfg": _canonicalize_cfg(preprocessing_cfg),
        "preprocessing_hash": compute_preprocessing_hash(preprocessing_cfg),
        "timestamp": datetime.utcnow().isoformat(),
        "generator_version": generator_version,
    }
    if extra:
        meta.update(extra)
    return meta



